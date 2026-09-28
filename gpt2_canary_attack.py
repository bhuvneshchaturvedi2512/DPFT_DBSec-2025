#!/usr/bin/env python3

import argparse
import math
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the prompt-based data extraction attack "
            "from Du et al., using a LoRA or fully fine-tuned GPT-2."
        )
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        help=(
            "Directory containing either a LoRA adapter or "
            "a fully fine-tuned Hugging Face model."
        ),
    )

    parser.add_argument(
        "--prefix",
        required=True,
        help=(
            "Partial canary prefix P used for both candidate "
            "generation and candidate scoring."
        ),
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=1000,
        help="Number C of UNIQUE candidate strings. Default: 1000.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        required=True,
        help=(
            "Nucleus sampling probability. "
            "Set to 0 to disable top-p."
        ),
    )

    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help=(
            "Top-k sampling value. "
            "Set to 0 to disable top-k."
        ),
    )

    parser.add_argument(
        "--length",
        type=int,
        required=True,
        help=(
            "Candidate truncation length in characters."
        ),
    )

    parser.add_argument(
        "--canary",
        required=True,
        help=(
            "Known true canary suffix S. "
            "It must have the same truncated length as candidates."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Plain-text output file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=(
            "Batch size for generation and scoring. "
            "Default: 32."
        ),
    )

    parser.add_argument(
        "--max_generation_batches",
        type=int,
        default=10000,
        help=(
            "Maximum number of generation batches allowed "
            "while collecting unique candidates."
        ),
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir):
    """
    Load either:

      1. A LoRA fine-tuned model:
           model_dir/
               adapter_config.json
               adapter_model.safetensors
               ...

         In this case, the base model is loaded first and
         the LoRA adapter is attached.

      2. A fully fine-tuned model:
           model_dir/
               config.json
               model.safetensors
               ...

         In this case, the model is loaded directly.

    The function returns the same model interface in both cases.
    """

    adapter_config_path = os.path.join(
        model_dir,
        "adapter_config.json",
    )

    is_lora = os.path.isfile(
        adapter_config_path
    )

    print("=" * 70)
    print("Loading model")
    print("=" * 70)

    # ============================================================
    # LoRA model
    # ============================================================

    if is_lora:

        peft_config = PeftConfig.from_pretrained(
            model_dir
        )

        base_model_name = (
            peft_config.base_model_name_or_path
        )

        print(
            "Model type:   LoRA fine-tuned"
        )

        print(
            f"Base model:   {base_model_name}"
        )

        print(
            f"LoRA adapter: {model_dir}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = (
            AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
            )
        )

        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            is_trainable=False,
        )

        print(
            f"Loaded LoRA adapter from: {model_dir}"
        )

        print(
            f"LoRA modules loaded: "
            f"{len(model.peft_config)}"
        )

    # ============================================================
    # Full fine-tuned model
    # ============================================================

    else:

        print(
            "Model type:   Fully fine-tuned"
        )

        print(
            f"Model:        {model_dir}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
        )

        print(
            f"Loaded fully fine-tuned model from: "
            f"{model_dir}"
        )

    # ============================================================
    # Common model setup
    # ============================================================

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
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    model.eval()

    if torch.cuda.is_available():

        device = torch.device("cuda")

        model = model.to(device)

    else:

        device = torch.device("cpu")

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    print(
        f"Device:       {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU:          "
            f"{torch.cuda.get_device_name(device)}"
        )

    print()

    return model, tokenizer, device


def generate_candidates_batch(
    model,
    tokenizer,
    device,
    prefix,
    batch_size,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
):
    """
    Generate one batch of candidate suffixes.

    The same prefix P is supplied independently to
    every candidate generation.

    Decoding:

      top_p == 0 and top_k == 0 -> greedy
      top_p > 0                -> nucleus sampling
      top_p == 0 and top_k > 0 -> top-k sampling
    """

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = inputs["input_ids"].to(device)

    attention_mask = inputs[
        "attention_mask"
    ].to(device)

    input_ids = input_ids.repeat(
        batch_size,
        1,
    )

    attention_mask = attention_mask.repeat(
        batch_size,
        1,
    )

    if top_p == 0 and top_k == 0:

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elif top_p > 0:

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elif top_k > 0:

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    else:

        raise ValueError(
            "Invalid decoding configuration."
        )

    prefix_length = input_ids.shape[1]

    generated_ids = outputs[
        :,
        prefix_length:
    ]

    candidates = []

    for ids in generated_ids:

        text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
        )

        candidates.append(text)

    return candidates


def collect_unique_candidates(
    model,
    tokenizer,
    device,
    prefix,
    candidate_count,
    temperature,
    top_p,
    top_k,
    length,
    batch_size,
    max_generation_batches,
):
    """
    Generate candidates until exactly C UNIQUE
    strings have been obtained.

    Truncation to l characters happens BEFORE
    uniqueness is checked.
    """

    unique_candidates = set()

    max_new_tokens = max(
        16,
        length * 4,
    )

    generation_batch = 0

    print(
        f"Collecting {candidate_count} unique "
        f"candidates..."
    )

    while (
        len(unique_candidates)
        < candidate_count
    ):

        generation_batch += 1

        if (
            generation_batch
            > max_generation_batches
        ):
            raise RuntimeError(
                "Could not collect the requested number "
                "of unique candidates.\n"
                f"Collected: {len(unique_candidates)}\n"
                f"Requested: {candidate_count}\n"
                f"Generation batches: "
                f"{generation_batch - 1}\n\n"
                "Consider increasing the sampling "
                "diversity or max_generation_batches."
            )

        remaining = (
            candidate_count
            - len(unique_candidates)
        )

        current_batch_size = min(
            batch_size,
            remaining,
        )

        generated = generate_candidates_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prefix=prefix,
            batch_size=current_batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )

        before = len(unique_candidates)

        for candidate in generated:

            candidate = candidate[:length]

            unique_candidates.add(candidate)

            if (
                len(unique_candidates)
                >= candidate_count
            ):
                break

        added = (
            len(unique_candidates)
            - before
        )

        print(
            f"  Batch {generation_batch}: "
            f"+{added} unique, "
            f"total "
            f"{len(unique_candidates)}/"
            f"{candidate_count}"
        )

    return list(unique_candidates)


def compute_candidate_ce_batch(
    model,
    tokenizer,
    device,
    prefix,
    candidates,
):
    """
    Compute CE(s | P) for each candidate.

    P is the generation/scoring context.

    Only candidate tokens contribute to the
    cross-entropy loss. Prefix tokens provide
    context but do not contribute to the loss.
    """

    batch_size = len(candidates)

    # ---------------------------------------------------------------
    # Tokenize prefix P
    # ---------------------------------------------------------------

    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    prefix_length = (
        prefix_ids.shape[0]
    )

    # ---------------------------------------------------------------
    # Tokenize candidate suffixes
    # ---------------------------------------------------------------

    candidate_encodings = tokenizer(
        candidates,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    candidate_ids = (
        candidate_encodings["input_ids"]
    )

    candidate_attention_mask = (
        candidate_encodings["attention_mask"]
    )

    candidate_padded_length = (
        candidate_ids.shape[1]
    )

    # ---------------------------------------------------------------
    # Construct:
    #
    #       [P tokens][candidate tokens]
    # ---------------------------------------------------------------

    prefix_batch = (
        prefix_ids
        .unsqueeze(0)
        .expand(
            batch_size,
            -1,
        )
    )

    prefix_attention = torch.ones(
        (
            batch_size,
            prefix_length,
        ),
        dtype=torch.long,
    )

    input_ids = torch.cat(
        [
            prefix_batch,
            candidate_ids,
        ],
        dim=1,
    )

    attention_mask = torch.cat(
        [
            prefix_attention,
            candidate_attention_mask,
        ],
        dim=1,
    )

    input_ids = input_ids.to(device)

    attention_mask = (
        attention_mask.to(device)
    )

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs.logits

    # GPT-2:
    #
    # logits[:, t, :] predicts token t+1
    #
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]

    # ---------------------------------------------------------------
    # Mask ONLY candidate tokens.
    # ---------------------------------------------------------------

    loss_mask = torch.zeros_like(
        shifted_labels,
        dtype=torch.bool,
    )

    loss_mask[
        :,
        prefix_length - 1:
        prefix_length - 1
        + candidate_padded_length,
    ] = candidate_attention_mask.to(
        device
    ).bool()

    # ---------------------------------------------------------------
    # Token-level CE
    # ---------------------------------------------------------------

    vocab_size = (
        shifted_logits.shape[-1]
    )

    token_losses = (
        torch.nn.functional.cross_entropy(
            shifted_logits.reshape(
                -1,
                vocab_size,
            ),
            shifted_labels.reshape(-1),
            reduction="none",
        )
    )

    token_losses = token_losses.reshape(
        batch_size,
        -1,
    )

    token_losses = (
        token_losses
        * loss_mask.float()
    )

    token_counts = (
        loss_mask.sum(dim=1)
    )

    losses = (
        token_losses.sum(dim=1)
        / token_counts.clamp(min=1)
    )

    losses = torch.where(
        token_counts > 0,
        losses,
        torch.full_like(
            losses,
            float("inf"),
        ),
    )

    return losses.cpu().tolist()


def score_all_candidates(
    model,
    tokenizer,
    device,
    prefix,
    candidates,
    batch_size,
):
    """
    Compute CE(s | P) for every candidate.
    """

    results = []

    total = len(candidates)

    print()
    print(
        "Computing candidate cross-entropy..."
    )

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        batch_candidates = candidates[
            start:end
        ]

        losses = compute_candidate_ce_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prefix=prefix,
            candidates=batch_candidates,
        )

        for i, loss in enumerate(
            losses
        ):

            results.append(
                {
                    "generation_index": (
                        start + i
                    ),
                    "candidate": (
                        batch_candidates[i]
                    ),
                    "loss": loss,
                }
            )

        print(
            f"  Scored {end}/{total}"
        )

    return results


def assign_ranks(results):
    """
    Assign competition ranks according to
    ascending CE loss.

    Example:

        losses = [1.0, 1.0, 2.0, 3.0]

        ranks  = [1,   1,   3,   4]
    """

    sorted_indices = sorted(
        range(len(results)),
        key=lambda i: results[i]["loss"],
    )

    previous_loss = None
    rank = 1

    for position, index in enumerate(
        sorted_indices
    ):

        current_loss = (
            results[index]["loss"]
        )

        if position == 0:

            rank = 1

        elif current_loss != previous_loss:

            rank = position + 1

        results[index]["rank"] = rank

        previous_loss = current_loss


def compute_exposure(
    rank,
    candidate_count,
):
    """
    exposure =
        log2(|C|) - log2(rank)
    """

    return (
        math.log2(candidate_count)
        - math.log2(rank)
    )


def write_results(
    output_path,
    model_type,
    model_dir,
    prefix,
    canary,
    candidate_count,
    temperature,
    top_p,
    top_k,
    length,
    seed,
    batch_size,
    results,
    canary_result,
):
    """
    Write the complete attack result.
    """

    sorted_results = sorted(
        results,
        key=lambda x: (
            x["rank"],
            x["loss"],
            x["generation_index"],
        ),
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 120 + "\n")
        f.write(
            "GPT-2 Data Extraction Attack\n"
        )
        f.write("=" * 120 + "\n")

        f.write(
            f"Model type:         {model_type}\n"
        )

        f.write(
            f"Model directory:    {model_dir}\n"
        )

        f.write(
            f"Prefix P:            {prefix!r}\n"
        )

        f.write(
            f"True canary S:       {canary!r}\n"
        )

        f.write(
            f"Unique candidates:   {candidate_count}\n"
        )

        f.write(
            f"Candidate length:    "
            f"{length} characters\n"
        )

        f.write(
            f"Temperature:         {temperature}\n"
        )

        f.write(
            f"Top-p:               {top_p}\n"
        )

        f.write(
            f"Top-k:               {top_k}\n"
        )

        f.write(
            f"Seed:                {seed}\n"
        )

        f.write(
            f"Batch size:          {batch_size}\n"
        )

        if (
            top_p == 0
            and top_k == 0
        ):

            f.write(
                "Decoding:            Greedy\n"
            )

        elif top_p > 0:

            f.write(
                "Decoding:            Nucleus sampling\n"
            )

        else:

            f.write(
                "Decoding:            Top-k sampling\n"
            )

        f.write(
            "Scoring:             CE(s | P)\n"
        )

        f.write(
            "Candidate set:       "
            "Unique after truncation\n"
        )

        f.write("=" * 120 + "\n\n")

        if canary_result is not None:

            f.write(
                "TRUE CANARY RESULT\n"
            )

            f.write(
                "-" * 120 + "\n"
            )

            f.write(
                f"Candidate:           "
                f"{canary_result['candidate']!r}\n"
            )

            f.write(
                f"CE loss:             "
                f"{canary_result['loss']:.10f}\n"
            )

            f.write(
                f"Rank:                "
                f"{canary_result['rank']}\n"
            )

            f.write(
                f"Exposure:            "
                f"{canary_result['exposure']:.10f}\n"
            )

            f.write("\n")

        else:

            f.write(
                "TRUE CANARY RESULT\n"
            )

            f.write(
                "-" * 120 + "\n"
            )

            f.write(
                "The true canary was NOT generated "
                "among the unique candidates.\n"
            )

            f.write(
                "Exposure was therefore not computed.\n"
            )

            f.write("\n")

        f.write(
            "ALL CANDIDATES\n"
        )

        f.write(
            "-" * 120 + "\n"
        )

        f.write(
            f"{'Candidate':<70} "
            f"{'Loss':>14} "
            f"{'Rank':>8} "
            f"{'Exposure':>14} "
            f"Marker\n"
        )

        f.write("-" * 120 + "\n")

        for result in sorted_results:

            candidate = result[
                "candidate"
            ]

            loss = result[
                "loss"
            ]

            rank = result[
                "rank"
            ]

            exposure = result[
                "exposure"
            ]

            marker = (
                "←"
                if result["is_match"]
                else ""
            )

            f.write(
                f"{candidate:<70} "
                f"{loss:>14.8f} "
                f"{rank:>8d} "
                f"{exposure:>14.8f} "
                f"{marker}\n"
            )


def main():

    args = parse_args()

    # ---------------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------------

    if args.candidates <= 0:

        raise ValueError(
            "--candidates must be > 0."
        )

    if args.length <= 0:

        raise ValueError(
            "--length must be > 0."
        )

    if args.temperature <= 0:

        raise ValueError(
            "--temperature must be > 0."
        )

    if not (
        0 <= args.top_p <= 1
    ):

        raise ValueError(
            "--top_p must be in [0, 1]."
        )

    if args.top_k < 0:

        raise ValueError(
            "--top_k must be >= 0."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch_size must be > 0."
        )

    if (
        args.top_p == 0
        and args.top_k == 0
        and args.candidates > 1
    ):

        raise ValueError(
            "Greedy decoding produces the same "
            "candidate for every generation. "
            "It cannot produce multiple unique "
            "candidates. Use sampling for C > 1."
        )

    # ---------------------------------------------------------------
    # Seed
    # ---------------------------------------------------------------

    if args.seed is not None:

        set_seed(args.seed)

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    model, tokenizer, device = (
        load_model(
            args.model_dir
        )
    )

    # Determine model type for output.
    is_lora = os.path.isfile(
        os.path.join(
            args.model_dir,
            "adapter_config.json",
        )
    )

    model_type = (
        "LoRA fine-tuned"
        if is_lora
        else "Fully fine-tuned"
    )

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    print("=" * 70)
    print("Attack configuration")
    print("=" * 70)

    print(
        f"Model type:         {model_type}"
    )

    print(
        f"Model directory:    "
        f"{args.model_dir}"
    )

    print(
        f"Prefix P:            {args.prefix!r}"
    )

    print(
        f"True canary S:       {args.canary!r}"
    )

    print(
        f"Candidates C:        {args.candidates}"
    )

    print(
        f"Length l:            "
        f"{args.length} characters"
    )

    print(
        f"Temperature:         "
        f"{args.temperature}"
    )

    print(
        f"Top-p:               "
        f"{args.top_p}"
    )

    print(
        f"Top-k:               "
        f"{args.top_k}"
    )

    print(
        f"Seed:                "
        f"{args.seed}"
    )

    print(
        f"Batch size:          "
        f"{args.batch_size}"
    )

    if (
        args.top_p == 0
        and args.top_k == 0
    ):

        print(
            "Decoding:            Greedy"
        )

    elif args.top_p > 0:

        print(
            "Decoding:            Nucleus sampling"
        )

    else:

        print(
            "Decoding:            Top-k sampling"
        )

    print(
        "Scoring:             CE(s | P)"
    )

    print(
        "Uniqueness:          After "
        "character truncation"
    )

    print("=" * 70)
    print()

    # ---------------------------------------------------------------
    # Generate UNIQUE candidates
    # ---------------------------------------------------------------

    candidates = collect_unique_candidates(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefix=args.prefix,
        candidate_count=args.candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        length=args.length,
        batch_size=args.batch_size,
        max_generation_batches=(
            args.max_generation_batches
        ),
    )

    # ---------------------------------------------------------------
    # Score all candidates
    # ---------------------------------------------------------------

    results = score_all_candidates(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefix=args.prefix,
        candidates=candidates,
        batch_size=args.batch_size,
    )

    # ---------------------------------------------------------------
    # Determine whether the true canary is in C
    # ---------------------------------------------------------------

    for result in results:

        result["is_match"] = (
            result["candidate"]
            == args.canary
        )

    # ---------------------------------------------------------------
    # Rank candidates
    # ---------------------------------------------------------------

    print()
    print(
        "Ranking candidates..."
    )

    assign_ranks(results)

    # ---------------------------------------------------------------
    # Exposure
    # ---------------------------------------------------------------

    for result in results:

        result["exposure"] = (
            compute_exposure(
                result["rank"],
                args.candidates,
            )
        )

    # ---------------------------------------------------------------
    # Locate true canary
    # ---------------------------------------------------------------

    canary_results = [
        result
        for result in results
        if result["is_match"]
    ]

    canary_result = None

    if canary_results:

        canary_result = canary_results[0]

    # ---------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------

    output_parent = os.path.dirname(
        os.path.abspath(args.output)
    )

    os.makedirs(
        output_parent,
        exist_ok=True,
    )

    write_results(
        output_path=args.output,
        model_type=model_type,
        model_dir=args.model_dir,
        prefix=args.prefix,
        canary=args.canary,
        candidate_count=args.candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        length=args.length,
        seed=args.seed,
        batch_size=args.batch_size,
        results=results,
        canary_result=canary_result,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("Attack completed")
    print("=" * 70)

    print(
        f"Model type:         {model_type}"
    )

    print(
        f"Unique candidates:  "
        f"{len(results)}"
    )

    print(
        f"Output file:        "
        f"{args.output}"
    )

    if canary_result is not None:

        print(
            "Canary found:       Yes"
        )

        print(
            f"Canary loss:        "
            f"{canary_result['loss']:.10f}"
        )

        print(
            f"Canary rank:        "
            f"{canary_result['rank']}"
        )

        print(
            f"Exposure:           "
            f"{canary_result['exposure']:.10f}"
        )

    else:

        print(
            "Canary found:       No"
        )

        print(
            "Exposure:           Not computed"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
