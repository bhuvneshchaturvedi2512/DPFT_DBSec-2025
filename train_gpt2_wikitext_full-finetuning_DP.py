#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from fastDP import PrivacyEngine


# ============================================================
# Defaults
# ============================================================

MODEL_NAME = "openai-community/gpt2"

MAX_LENGTH = 1024

TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16

NUM_EPOCHS = 10
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03

MAX_GRAD_NORM = 2.0

DELTA = 1e-5

EARLY_STOPPING_PATIENCE = 2

SEED = 42


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full GPT-2 fine-tuning with DP-Adam using fastDP."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the WikiText-2 DatasetDict.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory in which the final DP model is saved.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Target epsilon.",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=DELTA,
        help=f"Target delta. Default: {DELTA}",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help=f"Base model. Default: {MODEL_NAME}",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs. Default: {NUM_EPOCHS}",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate. Default: {LEARNING_RATE}",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=MAX_GRAD_NORM,
        help=f"Maximum per-sample gradient norm. Default: {MAX_GRAD_NORM}",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help=f"Physical training batch size. Default: {TRAIN_BATCH_SIZE}",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps. Default: {GRADIENT_ACCUMULATION_STEPS}",
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help=(
            "Optional limit on optimizer updates. "
            "Useful for a short smoke test. -1 means full training."
        ),
    )

    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Disable early stopping.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0.",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )
    
    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_all_seeds(seed):
    set_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset preparation
# ============================================================

def tokenize_record(example, tokenizer):
    text = example["text"]

    if text is None or not text.strip():
        text = tokenizer.eos_token

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH - 1,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"]

    # Ensure at least two tokens so GPT-2 has one causal-LM target.
    if len(input_ids) < 2:
        input_ids = input_ids + [tokenizer.eos_token_id]

    input_ids = input_ids[:MAX_LENGTH]

    attention_mask = [1] * len(input_ids)

    padding_length = MAX_LENGTH - len(input_ids)

    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length

    labels = input_ids.copy()

    for i in range(MAX_LENGTH):
        if attention_mask[i] == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_dataset(dataset, tokenizer):
    if "text" not in dataset.column_names:
        raise ValueError(
            f"Expected a 'text' column, found: {dataset.column_names}"
        )

    tokenized = dataset.map(
        lambda example: tokenize_record(example, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing records",
    )

    return tokenized


# ============================================================
# DataLoader
# ============================================================

def collate_batch(features):
    return {
        "input_ids": torch.tensor(
            [x["input_ids"] for x in features],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [x["attention_mask"] for x in features],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [x["labels"] for x in features],
            dtype=torch.long,
        ),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss encountered: {loss.item()}"
            )

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Validation dataloader is empty.")

    return total_loss / total_batches


# ============================================================
# Helper function for traching gradients
# ============================================================

def compute_grad_norms(model):
    """Computes total L2 norm of the model gradients."""
    total_sq_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_sq_norm += param_norm.item() ** 2
    return total_sq_norm ** 0.5

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.epsilon <= 0:
        raise ValueError("--epsilon must be > 0.")

    if args.delta <= 0 or args.delta >= 1:
        raise ValueError("--delta must satisfy 0 < delta < 1.")

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")

    if args.train_batch_size <= 0:
        raise ValueError("--train_batch_size must be > 0.")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be > 0.")

    os.makedirs(args.output_dir, exist_ok=True)

    set_all_seeds(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("DP GPT-2 Fine-Tuning with fastDP")
    print("=" * 70)
    print(f"Model:                       {args.model_name}")
    print(f"Dataset:                     {args.dataset_dir}")
    print(f"Output directory:            {args.output_dir}")
    print(f"Device:                      {device}")
    print(f"Epsilon:                     {args.epsilon}")
    print(f"Delta:                       {args.delta}")
    print(f"Max gradient norm:           {args.max_grad_norm}")
    print(f"Physical batch size:         {args.train_batch_size}")
    print(f"Gradient accumulation:      {args.gradient_accumulation_steps}")
    print(
        f"Effective logical batch:     "
        f"{args.train_batch_size * args.gradient_accumulation_steps}"
    )
    print(f"Epochs:                      {args.epochs}")
    print(f"Learning rate:               {args.learning_rate}")
    print(f"Weight decay:                {WEIGHT_DECAY}")
    print(f"Warmup ratio:                {WARMUP_RATIO}")
    print(f"Sequence length:             {MAX_LENGTH}")
    print(f"Seed:                        {args.seed}")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU:                         {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory:                  "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GB"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_from_disk(args.dataset_dir)

    if "train" not in dataset:
        raise ValueError(
            f"Dataset must contain a 'train' split. "
            f"Available splits: {list(dataset.keys())}"
        )

    if "validation" not in dataset:
        raise ValueError(
            f"Dataset must contain a 'validation' split. "
            f"Available splits: {list(dataset.keys())}"
        )

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print(f"Train records:                {len(train_dataset)}")
    print(f"Validation records:           {len(validation_dataset)}")

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocabulary size:    {len(tokenizer)}")
    print(f"Pad token:                    {tokenizer.pad_token}")
    print(f"Pad token ID:                 {tokenizer.pad_token_id}")

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print("\nTokenizing training records...")

    tokenized_train = prepare_dataset(
        train_dataset,
        tokenizer,
    )

    print("Tokenizing validation records...")

    tokenized_validation = prepare_dataset(
        validation_dataset,
        tokenizer,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    logical_batch_size = (
        args.train_batch_size *
        args.gradient_accumulation_steps
    )

    # Each optimizer update must contain exactly the logical
    # batch size assumed by fastDP's privacy accountant.
    #
    # Therefore:
    #   physical batch = 4
    #   accumulation    = 4
    #   logical batch   = 16
    #
    # Dropping the final incomplete logical batch keeps this
    # assumption fixed.

    train_loader = DataLoader(
        tokenized_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        tokenized_validation,
        batch_size=args.train_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    batches_per_epoch = len(train_loader)

    optimizer_steps_per_epoch = (
        batches_per_epoch // args.gradient_accumulation_steps
    )

    if optimizer_steps_per_epoch == 0:
        raise RuntimeError(
            "The training dataset is too small for the selected "
            "batch size and gradient accumulation."
        )

    total_optimizer_steps = (
        optimizer_steps_per_epoch * args.epochs
    )

    if args.max_train_steps > 0:
        total_optimizer_steps = min(
            total_optimizer_steps,
            args.max_train_steps,
        )

    print("\nTraining schedule:")
    print(f"Physical batches/epoch:       {batches_per_epoch}")
    print(f"Optimizer steps/epoch:        {optimizer_steps_per_epoch}")
    print(f"Total optimizer steps:        {total_optimizer_steps}")

    # --------------------------------------------------------
    # Load GPT-2
    # --------------------------------------------------------

    print("\nLoading GPT-2...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Full fine-tuning.
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Total parameters:             {total_params:,}")
    print(f"Trainable parameters:         {trainable_params:,}")

    if trainable_params != total_params:
        raise RuntimeError(
            "The model is not configured for full fine-tuning."
        )

    model.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    # fastDP supports standard PyTorch optimizers. We use Adam
    # here because this experiment is intended to use DP-Adam.
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    num_warmup_steps = int(
        WARMUP_RATIO * total_optimizer_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    print(f"Warmup steps:                 {num_warmup_steps}")

    # --------------------------------------------------------
    # fastDP PrivacyEngine
    # --------------------------------------------------------

    print("\nCreating fastDP PrivacyEngine...")

    privacy_engine = PrivacyEngine(
        model,
        batch_size=logical_batch_size,
        sample_size=len(tokenized_train),
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        target_epsilon=args.epsilon,
        target_delta=args.delta,

        # Book-Keeping / ghost clipping.
        clipping_mode="ghost",

        # Explicit fixed clipping threshold.
        clipping_fn="global",

        # Clip the complete model gradient.
        clipping_style="all-layer",

        # GPT-2 ghost differentiation origin parameters.
        origin_params=["wte", "wpe"],

        # RDP accounting.
        accounting_mode="rdp",

        # Model loss is the usual mean language-model loss.
        loss_reduction="mean",
    )

    print("\nfastDP configuration:")
    print(privacy_engine)

    print(
        f"\nNoise multiplier:             "
        f"{privacy_engine.noise_multiplier:.8f}"
    )

    print(
        f"Effective noise multiplier:   "
        f"{privacy_engine.effective_noise_multiplier:.8f}"
    )

    print(
        f"Target epsilon:               "
        f"{privacy_engine.target_epsilon}"
    )

    print(
        f"Target delta:                 "
        f"{privacy_engine.target_delta}"
    )

    print(
        f"Sample rate:                  "
        f"{privacy_engine.sample_rate:.10f}"
    )

    privacy_engine.attach(optimizer)

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config_to_save = {
        "model_name": args.model_name,
        "dataset_dir": args.dataset_dir,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "max_grad_norm": args.max_grad_norm,
        "clipping_mode": "ghost",
        "clipping_fn": "global",
        "clipping_style": "all-layer",
        "origin_params": ["wte", "wpe"],
        "accounting_mode": "rdp",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "epochs": args.epochs,
        "physical_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logical_batch_size": logical_batch_size,
        "sequence_length": MAX_LENGTH,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": num_warmup_steps,
        "seed": args.seed,
        "noise_multiplier": privacy_engine.noise_multiplier,
        "effective_noise_multiplier": (
            privacy_engine.effective_noise_multiplier
        ),
        "sample_rate": privacy_engine.sample_rate,
        "train_records": len(tokenized_train),
        "validation_records": len(tokenized_validation),
    }

    with open(
        os.path.join(args.output_dir, "dp_config.json"),
        "w",
    ) as f:
        json.dump(config_to_save, f, indent=2)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    best_state_path = os.path.join(
        args.output_dir,
        "best_model_state.pt",
    )

    global_step = 0

    print("\n" + "=" * 70)
    print("Starting DP fine-tuning")
    print("=" * 70)

    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):

        if global_step >= total_optimizer_steps:
            break

        model.train()

        epoch_loss = 0.0
        epoch_microsteps = 0
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(train_loader):

            if global_step >= total_optimizer_steps:
                break

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            
            if not torch.isfinite(input_ids.float()).all():
                raise RuntimeError(f"Non-finite input_ids at batch {batch_idx}")

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            
            if not torch.isfinite(attention_mask.float()).all():
                raise RuntimeError(f"Non-finite attention_mask at batch {batch_idx}")

            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )
            
            if not torch.isfinite(labels.float()).all():
                raise RuntimeError(f"Non-finite labels at batch {batch_idx}")

            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    raise RuntimeError(f"Non-finite parameter BEFORE forward at epoch={epoch + 1}, batch={batch_idx}: {name}")
            
            valid_tokens = (labels != -100).sum(dim=1)

            if batch_idx >= 18 and batch_idx <= 22:
                print(f"  batch {batch_idx}: valid tokens per record = {valid_tokens.tolist()}, total valid tokens = {valid_tokens.sum().item()}")
                
            shifted_valid_tokens = (labels[:, 1:] != -100).sum(dim=1)

            if batch_idx >= 18 and batch_idx <= 22:
                print(f"  batch {batch_idx}: shifted valid tokens per record = {shifted_valid_tokens.tolist()}, total shifted valid tokens = {shifted_valid_tokens.sum().item()}")
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

            loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at "
                    f"epoch={epoch + 1}, batch={batch_idx}: "
                    f"{loss.item()}"
                )

            epoch_loss += loss.item()
            epoch_microsteps += 1

            # IMPORTANT:
            # Do not divide the loss by gradient_accumulation_steps.
            #
            # fastDP clips each individual sample contribution.
            # The private gradients are accumulated across the
            # microbatches and normalized using the logical batch
            # size supplied to PrivacyEngine.

            loss.backward()

            is_update_step = (
                (batch_idx + 1) % args.gradient_accumulation_steps == 0
            )

            if is_update_step:
                # 1. Compute effective post-DP gradient norm across all model parameters
                post_dp_grad_norm = compute_grad_norms(model)

                # 2. Extract fastDP per-sample clipping factor (if available)
                clipping_factor = getattr(privacy_engine, "norm_clipper", None)
                
                optimizer.step()
                for name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        raise RuntimeError(f"Non-finite parameter after DP optimizer step: {name}")
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % 10 == 0 or global_step == 1:
                    privacy = privacy_engine.get_privacy_spent()

                    print(
                        f"  step {global_step:6d} | "
                        f"loss {loss.item():.5f} | "
                        f"lr {scheduler.get_last_lr()[0]:.3e} | "
                        f"epsilon {privacy.get('eps', float('nan')):.5f}"
                    )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_loss = evaluate(
            model,
            validation_loader,
            device,
        )

        epoch_train_loss = (
            epoch_loss / max(epoch_microsteps, 1)
        )

        privacy = privacy_engine.get_privacy_spent()

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1} completed in "
            f"{epoch_time / 60.0:.2f} min"
        )

        print(
            f"  Train loss:                  "
            f"{epoch_train_loss:.6f}"
        )

        print(
            f"  Validation loss:             "
            f"{validation_loss:.6f}"
        )

        print(
            f"  Privacy epsilon:             "
            f"{privacy.get('eps', float('nan')):.8f}"
        )

        print(
            f"  Privacy delta:               "
            f"{privacy_engine.target_delta:.8e}"
        )

        # ----------------------------------------------------
        # Best-model tracking
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save only the model parameters.
            torch.save(
                model.state_dict(),
                best_state_path,
            )

            print(
                f"  New best validation loss. "
                f"Saved epoch {best_epoch}."
            )

        else:
            epochs_without_improvement += 1

            print(
                f"  No validation improvement. "
                f"Patience: {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            not args.no_early_stopping
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            print("  Early stopping triggered.")
            break

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if os.path.exists(best_state_path):

        print("\nRestoring best model...")

        state_dict = torch.load(
            best_state_path,
            map_location=device,
        )

        model.load_state_dict(state_dict)

    # --------------------------------------------------------
    # Final privacy information
    # --------------------------------------------------------

    final_privacy = privacy_engine.get_privacy_spent()

    print("\n" + "=" * 70)
    print("Training completed")
    print("=" * 70)

    print(
        f"Total training time:          "
        f"{total_time / 3600.0:.2f} hours"
    )

    print(
        f"Best epoch:                   "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss:         "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Optimizer steps:              "
        f"{privacy_engine.steps}"
    )

    print(
        f"Final epsilon:                "
        f"{final_privacy.get('eps', float('nan')):.8f}"
    )

    print(
        f"Final delta:                  "
        f"{privacy_engine.target_delta:.8e}"
    )

    print(
        f"Configured target epsilon:    "
        f"{args.epsilon}"
    )

    print(
        f"Configured target delta:      "
        f"{args.delta}"
    )

    # --------------------------------------------------------
    # Detach fastDP before final serialization
    # --------------------------------------------------------

    privacy_engine.detach()

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    print("\nSaving final model...")

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(args.output_dir)

    # Save final training information.
    final_info = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "optimizer_steps": privacy_engine.steps,
        "final_epsilon": final_privacy.get(
            "eps",
            None,
        ),
        "final_delta": privacy_engine.target_delta,
        "target_epsilon": args.epsilon,
        "target_delta": args.delta,
        "noise_multiplier": privacy_engine.noise_multiplier,
        "training_time_seconds": total_time,
    }

    with open(
        os.path.join(args.output_dir, "training_results.json"),
        "w",
    ) as f:
        json.dump(final_info, f, indent=2)

    # Remove temporary state file.
    if os.path.exists(best_state_path):
        os.remove(best_state_path)

    print("\nFinal model saved to:")
    print(f"  {args.output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
