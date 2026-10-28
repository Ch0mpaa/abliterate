#!/usr/bin/env python3
"""
Compute a model's refusal direction from contrasting harmful/harmless prompt pairs.

This is the correct way to find the refusal direction — NOT SVD on weight matrices.
SVD extracts dominant weight structure (general capability), not behavioral directions.
Contrasting pairs measure where the model's activations diverge when processing
harmful vs harmless prompts, giving a precise refusal-specific direction.

Method (following huginnfork's approach):
1. Load harmful prompts (mlabonne/harmful_behaviors) and harmless prompts (mlabonne/harmless_alpaca)
2. Run each prompt through the model, record residual stream activations at every layer
3. Compute mean(harmful_activations) - mean(harmless_activations) per layer
4. Select the layer with the highest Cohen's d (effect size)
5. Normalize and save the direction vector

The output .npz is compatible with all abliteration scripts via --direction.

Usage:
    python3 compute_direction.py --model /path/to/model --output direction.npz [--n-prompts 128]

    # Then abliterate with the computed direction:
    python3 abliterate_bf16.py --model /path/to/model --output ./abliterated --direction direction.npz

Requirements:
    pip install torch transformers datasets numpy

GPU recommended for large models. CPU works for models up to ~7B.
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch


def load_prompts(n_harmful: int, n_harmless: int, seed: int = 42):
    """Load harmful and harmless prompt sets."""
    from datasets import load_dataset

    print(f"Loading {n_harmful} harmful + {n_harmless} harmless prompts...")

    harmful_ds = load_dataset("mlabonne/harmful_behaviors", split="train")
    harmless_ds = load_dataset("mlabonne/harmless_alpaca", split="train")

    rng = np.random.default_rng(seed)

    h_indices = rng.choice(len(harmful_ds), size=min(n_harmful, len(harmful_ds)), replace=False)
    s_indices = rng.choice(len(harmless_ds), size=min(n_harmless, len(harmless_ds)), replace=False)

    harmful_prompts = []
    for i in h_indices:
        row = harmful_ds[int(i)]
        text = row.get("text") or row.get("prompt") or row.get("instruction") or str(row)
        harmful_prompts.append(text)

    harmless_prompts = []
    for i in s_indices:
        row = harmless_ds[int(i)]
        text = row.get("text") or row.get("prompt") or row.get("instruction") or str(row)
        harmless_prompts.append(text)

    print(f"  Loaded {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless")
    return harmful_prompts, harmless_prompts


def collect_activations(model, tokenizer, prompts: list, device: str, max_length: int = 256):
    """Run prompts through model and collect residual stream activations at each layer."""
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    all_activations = [[] for _ in range(num_layers)]

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

        for layer_idx in range(num_layers):
            # Take the mean across the sequence dimension for this layer's output
            # hidden_states[0] is embeddings, [1] is after layer 0, etc.
            h = hidden_states[layer_idx + 1]  # (1, seq_len, hidden_size)
            h_mean = h.mean(dim=1).squeeze(0).float().cpu().numpy()  # (hidden_size,)
            all_activations[layer_idx].append(h_mean)

        if (i + 1) % 16 == 0 or i == 0:
            print(f"  [{i+1}/{len(prompts)}] collected")

    # Stack into arrays: (n_prompts, hidden_size) per layer
    stacked = [np.stack(acts) for acts in all_activations]
    return stacked


def compute_direction(harmful_acts: list, harmless_acts: list, hidden_size: int,
                      mask_massive: bool = True):
    """
    Compute refusal direction per layer and select the best one.

    Returns:
        direction: (hidden_size,) normalized direction vector
        layer: index of the selected layer
        cohen_d: effect size at the selected layer
        all_stats: dict with per-layer statistics
    """
    num_layers = len(harmful_acts)
    all_dirs = np.zeros((num_layers, hidden_size), dtype=np.float32)
    all_cohen_d = np.zeros(num_layers, dtype=np.float32)
    all_diff_norms = np.zeros(num_layers, dtype=np.float32)
    all_harmful_means = np.zeros((num_layers, hidden_size), dtype=np.float32)
    all_harmless_means = np.zeros((num_layers, hidden_size), dtype=np.float32)
    masked_dims = np.zeros(num_layers, dtype=np.int32)

    for layer_idx in range(num_layers):
        h_mean = harmful_acts[layer_idx].mean(axis=0)   # (hidden_size,)
        s_mean = harmless_acts[layer_idx].mean(axis=0)   # (hidden_size,)

        diff = h_mean - s_mean

        if mask_massive:
            # Mask dimensions with extremely large values that would dominate
            abs_diff = np.abs(diff)
            threshold = np.median(abs_diff) + 10 * np.std(abs_diff)
            massive = abs_diff > threshold
            n_masked = massive.sum()
            if n_masked > 0 and n_masked < hidden_size // 2:
                diff[massive] = 0.0
                masked_dims[layer_idx] = n_masked

        diff_norm = np.linalg.norm(diff)
        if diff_norm > 1e-10:
            direction = diff / diff_norm
        else:
            direction = diff

        # Cohen's d: effect size between harmful and harmless distributions
        # Project each sample onto the direction, compute mean and pooled std
        h_proj = harmful_acts[layer_idx] @ direction   # (n_harmful,)
        s_proj = harmless_acts[layer_idx] @ direction   # (n_harmless,)

        h_var = h_proj.var(ddof=1) if len(h_proj) > 1 else 1e-10
        s_var = s_proj.var(ddof=1) if len(s_proj) > 1 else 1e-10
        pooled_std = np.sqrt((h_var + s_var) / 2)

        if pooled_std > 1e-10:
            d = (h_proj.mean() - s_proj.mean()) / pooled_std
        else:
            d = 0.0

        all_dirs[layer_idx] = direction
        all_cohen_d[layer_idx] = abs(d)
        all_diff_norms[layer_idx] = diff_norm
        all_harmful_means[layer_idx] = h_mean
        all_harmless_means[layer_idx] = s_mean

    # Select layer with highest Cohen's d
    best_layer = int(np.argmax(all_cohen_d))
    best_d = all_cohen_d[best_layer]
    best_dir = all_dirs[best_layer]

    print(f"\n  Layer selection:")
    print(f"  Best layer: {best_layer}/{num_layers} (Cohen's d = {best_d:.2f})")
    print(f"  Diff norm: {all_diff_norms[best_layer]:.4f}")
    print(f"  Masked dims: {masked_dims[best_layer]}")

    # Show top 5 layers
    top5 = np.argsort(all_cohen_d)[-5:][::-1]
    for rank, idx in enumerate(top5):
        print(f"  #{rank+1}: layer {idx} d={all_cohen_d[idx]:.2f} norm={all_diff_norms[idx]:.4f}")

    return best_dir, best_layer, best_d, {
        "all_dirs": all_dirs,
        "all_cohen_d": all_cohen_d,
        "all_diff_norms": all_diff_norms,
        "all_harmful_means": all_harmful_means,
        "all_harmless_means": all_harmless_means,
        "masked_dims_per_layer": masked_dims,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute refusal direction from contrasting harmful/harmless prompts")
    parser.add_argument("--model", required=True, help="Path to model (local or HuggingFace ID)")
    parser.add_argument("--output", required=True, help="Output .npz file path")
    parser.add_argument("--n-prompts", type=int, default=128,
                        help="Number of harmful AND harmless prompts (default: 128)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length per prompt (default: 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mask", action="store_true",
                        help="Don't mask massive outlier dimensions")
    parser.add_argument("--dtype", default="auto",
                        help="Model dtype: auto, float16, bfloat16, float32")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Dtype selection
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": None,
    }
    dtype = dtype_map.get(args.dtype)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True, "output_hidden_states": True}
    if dtype:
        load_kwargs["torch_dtype"] = dtype
    if device == "cpu":
        load_kwargs["device_map"] = "cpu"
        if not dtype:
            load_kwargs["torch_dtype"] = torch.float32
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    print(f"Loaded in {time.time()-t0:.1f}s — {num_layers} layers, hidden_size={hidden_size}")

    # Load prompts
    harmful_prompts, harmless_prompts = load_prompts(args.n_prompts, args.n_prompts, args.seed)

    # Collect activations
    print(f"\nCollecting harmful activations...")
    harmful_acts = collect_activations(model, tokenizer, harmful_prompts, device, args.max_length)

    print(f"\nCollecting harmless activations...")
    harmless_acts = collect_activations(model, tokenizer, harmless_prompts, device, args.max_length)

    # Compute direction
    print(f"\nComputing refusal direction...")
    direction, layer, cohen_d, stats = compute_direction(
        harmful_acts, harmless_acts, hidden_size, mask_massive=not args.no_mask)

    # Compute hashes for reproducibility
    harmful_sha = hashlib.sha256(str(harmful_prompts).encode()).hexdigest()
    harmless_sha = hashlib.sha256(str(harmless_prompts).encode()).hexdigest()

    # Save — compatible format with huginnfork's direction files
    np.savez(
        str(output_path),
        # Primary direction vector — this is what abliteration scripts use
        v=direction.astype(np.float32),
        # Selected layer info
        layer=np.array(layer),
        cohen_d=np.array(cohen_d, dtype=np.float32),
        diff_norm=np.array(stats["all_diff_norms"][layer], dtype=np.float32),
        # Per-layer data for analysis
        all_dirs=stats["all_dirs"],
        all_cohen_d=stats["all_cohen_d"],
        all_diff_norms=stats["all_diff_norms"],
        all_harmless_means=stats["all_harmless_means"],
        masked_dims_per_layer=stats["masked_dims_per_layer"],
        # Activation means at selected layer
        harmful_mean=stats["all_harmful_means"][layer],
        harmless_mean=stats["all_harmless_means"][layer],
        # Metadata for reproducibility
        hidden=np.array(hidden_size),
        num_layers=np.array(num_layers),
        num_harmful=np.array(len(harmful_prompts)),
        num_harmless=np.array(len(harmless_prompts)),
        seed=np.array(args.seed),
        enable_thinking=np.array(0),
        mask_massive=np.array(0 if args.no_mask else 1),
        harmful_repo=np.array("mlabonne/harmful_behaviors"),
        harmless_repo=np.array("mlabonne/harmless_alpaca"),
        harmful_sha256=np.array(harmful_sha),
        harmless_sha256=np.array(harmless_sha),
    )

    print(f"\n{'='*60}")
    print(f"REFUSAL DIRECTION COMPUTED")
    print(f"  Model: {args.model}")
    print(f"  Best layer: {layer}/{num_layers}")
    print(f"  Cohen's d: {cohen_d:.2f}")
    print(f"  Direction norm: 1.0 (normalized)")
    print(f"  Hidden dim: {hidden_size}")
    print(f"  Prompts: {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless")
    print(f"  Output: {output_path}")
    print(f"")
    print(f"  Next: abliterate with this direction:")
    print(f"    python3 abliterate_bf16.py --model {args.model} \\")
    print(f"      --output ./abliterated --direction {output_path} --lambda 3.0")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
