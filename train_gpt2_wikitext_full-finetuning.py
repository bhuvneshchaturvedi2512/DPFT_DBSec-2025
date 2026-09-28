#!/usr/bin/env python3

import os
import json
import math
import random
import argparse

import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)


# ============================================================
# Constants
# ============================================================

MODEL_NAME = "openai-community/gpt2"

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-v1"

CANARY = "The secret code is hzdh0831"

CANARY_FRACTION = 0.0025

BLOCK_SIZE = 1024

# Training
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2

LEARNING_RATE = 2e-4
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Insert canaries into the training dataset
# ============================================================

def insert_canaries(dataset, seed):
    """
    Insert standalone canary records at random positions.

    The number of canaries is floor(0.25% of the original
    training-set size).

    Original records are not modified.
    Validation and test splits are not touched.
    """

    train_dataset = dataset["train"]

    original_size = len(train_dataset)

    num_canaries = int(
        original_size * CANARY_FRACTION
    )

    print("\nOriginal training records:", original_size)
    print("Canary fraction:", CANARY_FRACTION)
    print("Number of canary records:", num_canaries)

    # --------------------------------------------------------
    # Construct the final dataset size.
    #
    # There are N original records and K canary records.
    # We randomly select K positions among N+K positions.
    # --------------------------------------------------------

    final_size = original_size + num_canaries

    rng = random.Random(seed)

    canary_positions = set(
        rng.sample(
            range(final_size),
            num_canaries
        )
    )

    # --------------------------------------------------------
    # Construct the modified records.
    # --------------------------------------------------------

    modified_records = []

    original_index = 0

    for position in range(final_size):

        if position in canary_positions:
            modified_records.append(CANARY)
        else:
            modified_records.append(
                train_dataset[original_index]["text"]
            )
            original_index += 1

    assert original_index == original_size

    assert len(modified_records) == final_size

    # Verify the exact number of canaries.
    assert modified_records.count(CANARY) == num_canaries

    # --------------------------------------------------------
    # Create a new Dataset.
    # --------------------------------------------------------

    modified_train = train_dataset.from_dict(
        {
            "text": modified_records
        }
    )

    modified_dataset = DatasetDict(
        {
            "train": modified_train,
            "validation": dataset["validation"],
            "test": dataset["test"],
        }
    )

    return modified_dataset, num_canaries, canary_positions


# ============================================================
# Tokenization
# ============================================================

def tokenize_dataset(dataset, tokenizer):

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing WikiText-2",
    )

    return tokenized


# ============================================================
# Group tokens into fixed-length blocks
# ============================================================

def group_texts(examples):

    concatenated_examples = {
        key: sum(examples[key], [])
        for key in examples.keys()
    }

    total_length = len(
        concatenated_examples["input_ids"]
    )

    # Drop incomplete final block.
    total_length = (
        total_length // BLOCK_SIZE
    ) * BLOCK_SIZE

    result = {
        key: [
            t[i : i + BLOCK_SIZE]
            for i in range(
                0,
                total_length,
                BLOCK_SIZE
            )
        ]
        for key, t in concatenated_examples.items()
    }

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Full fine-tuning of GPT-2 on WikiText-2 "
            "with repeated canary records."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for canary placement.",
    )

    parser.add_argument(
        "--modified_dataset_dir",
        type=str,
        default="./wikitext2_canary",
        help="Directory for the modified WikiText-2 dataset.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gpt2-wikitext2-full-canary",
        help="Directory for full fine-tuning checkpoints and model.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print(f"GPT-2 Full Fine-Tuning on {DATASET_CONFIG}")
    print("=" * 70)

    print("\nModel:", MODEL_NAME)
    print("Dataset:", DATASET_NAME)
    print("Configuration:", DATASET_CONFIG)
    print("Canary:", repr(CANARY))
    print("Canary fraction:", CANARY_FRACTION)
    print("Maximum epochs:", NUM_EPOCHS)
    print("Early-stopping patience:", EARLY_STOPPING_PATIENCE)
    print("Learning rate:", LEARNING_RATE)
    print("Seed:", args.seed)

    # --------------------------------------------------------
    # 1. Load WikiText-2-v1
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"Loading {DATASET_CONFIG}")
    print("=" * 70)

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.expanduser(
                "~/wikitext-2-v1/train.parquet"
            ),
            "validation": os.path.expanduser(
                "~/wikitext-2-v1/validation.parquet"
            ),
            "test": os.path.expanduser(
                "~/wikitext-2-v1/test.parquet"
            ),
        },
    )

    print(dataset)

    assert len(dataset["train"]) == 36718, (
        f"Unexpected training-set size: "
        f"{len(dataset['train'])}. Expected 36718."
    )

    assert len(dataset["validation"]) == 3760, (
        f"Unexpected validation-set size: "
        f"{len(dataset['validation'])}. Expected 3760."
    )

    assert len(dataset["test"]) == 4358, (
        f"Unexpected test-set size: "
        f"{len(dataset['test'])}. Expected 4358."
    )

    print(f"\nVerified {DATASET_CONFIG} split sizes:")
    print("Train:      ", len(dataset["train"]))
    print("Validation: ", len(dataset["validation"]))
    print("Test:       ", len(dataset["test"]))

    # --------------------------------------------------------
    # 2. Insert canaries
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Creating modified dataset")
    print("=" * 70)

    modified_dataset, num_canaries, canary_positions = (
        insert_canaries(
            dataset,
            args.seed,
        )
    )

    print(
        "Modified training records:",
        len(modified_dataset["train"])
    )

    print(
        "Validation records:",
        len(modified_dataset["validation"])
    )

    print(
        "Test records:",
        len(modified_dataset["test"])
    )

    # --------------------------------------------------------
    # 3. Verify validation and test were not modified
    # --------------------------------------------------------

    assert (
        modified_dataset["validation"]
        == dataset["validation"]
    )

    assert (
        modified_dataset["test"]
        == dataset["test"]
    )

    # --------------------------------------------------------
    # 4. Save modified dataset separately
    # --------------------------------------------------------

    print("\nSaving modified dataset to:")
    print(args.modified_dataset_dir)

    modified_dataset.save_to_disk(
        args.modified_dataset_dir
    )

    # Save experiment metadata.
    metadata = {
        "base_dataset": DATASET_NAME,
        "base_config": DATASET_CONFIG,
        "original_train_size": len(dataset["train"]),
        "modified_train_size": len(
            modified_dataset["train"]
        ),
        "validation_size": len(
            modified_dataset["validation"]
        ),
        "test_size": len(
            modified_dataset["test"]
        ),
        "canary": CANARY,
        "canary_fraction": CANARY_FRACTION,
        "num_canaries": num_canaries,
        "seed": args.seed,
        "canary_positions": sorted(
            canary_positions
        ),
    }

    metadata_path = os.path.join(
        args.modified_dataset_dir,
        "canary_metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "Metadata saved to:",
        metadata_path
    )

    # --------------------------------------------------------
    # 5. Verify canary count
    # --------------------------------------------------------

    actual_canary_count = sum(
        1
        for record in modified_dataset["train"]
        if record["text"] == CANARY
    )

    print(
        "\nVerified canary records:",
        actual_canary_count
    )

    assert actual_canary_count == num_canaries

    # --------------------------------------------------------
    # 6. Load GPT-2 tokenizer
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Loading GPT-2 tokenizer")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # GPT-2 has no pad token by default.
    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # 7. Tokenize modified dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Tokenizing dataset")
    print("=" * 70)

    tokenized_dataset = tokenize_dataset(
        modified_dataset,
        tokenizer,
    )

    # --------------------------------------------------------
    # 8. Group tokens into 1024-token blocks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Creating language-modeling blocks")
    print("=" * 70)

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=(
            f"Grouping tokens into "
            f"{BLOCK_SIZE}-token blocks"
        ),
    )

    print("\nFinal LM dataset:")
    print(lm_dataset)

    # --------------------------------------------------------
    # 9. Load GPT-2
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Loading GPT-2 model")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # --------------------------------------------------------
    # 10. Verify full fine-tuning
    #
    # No PEFT/LoRA configuration is applied.
    # All model parameters remain trainable.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Configuring full fine-tuning")
    print("=" * 70)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters:     {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    assert (
        trainable_parameters == total_parameters
    ), (
        "Not all model parameters are trainable. "
        "Full fine-tuning was not configured correctly."
    )

    print("All model parameters are trainable.")

    # --------------------------------------------------------
    # 11. Data collator
    # --------------------------------------------------------

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --------------------------------------------------------
    # 12. Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Maximum number of epochs.
        # Early stopping can terminate training earlier.
        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        warmup_ratio=WARMUP_RATIO,

        weight_decay=WEIGHT_DECAY,

        logging_steps=100,

        # Evaluate after every epoch.
        eval_strategy="epoch",

        # Save after every epoch so that early stopping
        # can restore the best checkpoint.
        save_strategy="epoch",

        save_total_limit=2,

        # Restore the checkpoint with the lowest
        # validation loss.
        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        # Use FP16 when CUDA is available.
        fp16=torch.cuda.is_available(),

        report_to="none",

        seed=args.seed,

        data_seed=args.seed,
    )

    # --------------------------------------------------------
    # 13. Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=lm_dataset["train"],

        eval_dataset=lm_dataset["validation"],

        tokenizer=tokenizer,

        data_collator=data_collator,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],
    )

    # --------------------------------------------------------
    # 14. Full fine-tuning
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Starting full fine-tuning")
    print("=" * 70)

    trainer.train()

    # --------------------------------------------------------
    # 15. Final evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Final evaluation")
    print("=" * 70)

    eval_results = trainer.evaluate()

    print("\nEvaluation results:")

    for key, value in eval_results.items():
        print(f"{key}: {value}")

    if "eval_loss" in eval_results:
        perplexity = math.exp(
            eval_results["eval_loss"]
        )

        print(
            f"\nValidation perplexity: "
            f"{perplexity:.4f}"
        )

    # --------------------------------------------------------
    # 16. Save fully fine-tuned model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Saving fully fine-tuned model")
    print("=" * 70)

    trainer.save_model(
        args.output_dir
    )

    tokenizer.save_pretrained(
        args.output_dir
    )

    print(
        "\nFully fine-tuned model saved to:",
        args.output_dir
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
