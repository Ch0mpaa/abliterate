#!/usr/bin/env python3
"""
RedOps Stock NVFP4 Abliteration
================================
Applies pre-computed refusal direction to the stock Qwen3.8-Flash-Next NVFP4
checkpoint's BF16 layers. FP4 quantized weights are untouched.

Uses huginnfork's refusal direction (direction_off_a.npz key 'v') which was
computed from contrasting harmful/harmless prompts on the same base model.

The stock model's quant layout:
  - NVFP4 routed experts (quantized, NOT modified)
  - FP8_PB_WO MTP experts (quantized, NOT modified)
  - BF16 attention, embeddings, norms, lm_head, MTP non-expert (THESE get abliterated)

Output preserves exact same quant format → TensorFold 1.1.6 loads it.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Paths
HOME = Path.home()
STOCK_SNAP = HOME / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
DIRECTION_FILE = HOME / "direction_off_a.npz"
OUTPUT_DIR = HOME / "models/qwen38-flash-abliterated-nvfp4"

LAMBDA = float(os.environ.get("ABLIT_LAMBDA", "3.0"))
HIDDEN_DIM = 2560  # Qwen3.8-Flash-Next hidden size

# Layers to abliterate — BF16 attention/embed + FP8 shared experts
ABLITERATE_PATTERNS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
    "shared_expert_gate.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.layers.",
    "mtp.hyper_connection",
    "hyper_connection",
]

# Patterns to SKIP (quantized, scales, zeros, norms)
SKIP_PATTERNS = [
    "_scale", "_zero", "amax", "input_scale",
    "norm", "layernorm",
    "mlp.experts.",  # routed experts are FP4 — won't match shared_expert
]


def should_abliterate(key):
    key_lower = key.lower()
    if any(skip in key_lower for skip in SKIP_PATTERNS):
        return False
    if any(pat in key for pat in ABLITERATE_PATTERNS):
        return True
    return False


FP8_TYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16} | FP8_TYPES


def apply_abliteration(tensor, direction, per_layer_dir=None, lam=3.0):
    orig_dtype = tensor.dtype
    if orig_dtype not in SUPPORTED_DTYPES:
        return tensor
    w = tensor.float()
    d = direction.float()

    if w.dim() == 2:
        # Weight matrix: project out direction from rows or columns
        if w.shape[-1] == d.shape[0]:
            proj = torch.outer(w @ d, d)
            w = w - lam * proj
        elif w.shape[0] == d.shape[0]:
            proj = torch.outer(d, w.T @ d)
            w = w - lam * proj
    elif w.dim() == 1 and w.shape[0] == d.shape[0]:
        proj = (w @ d) * d
        w = w - lam * proj

    return w.to(orig_dtype)


def main():
    print("=" * 60)
    print("RedOps Stock NVFP4 Abliteration Pipeline")
    print("=" * 60)
    print(f"Stock model:  {STOCK_SNAP}")
    print(f"Direction:    {DIRECTION_FILE}")
    print(f"Output:       {OUTPUT_DIR}")
    print(f"Lambda:       {LAMBDA}")
    print()

    # Validate stock model
    tensors_dir = STOCK_SNAP / "tensors"
    if not tensors_dir.exists():
        print(f"ERROR: Stock tensors not found at {tensors_dir}")
        sys.exit(1)

    shard_files = sorted(tensors_dir.glob("model-*.safetensors"))
    print(f"Stock shards: {len(shard_files)}")

    # Load refusal direction
    if not DIRECTION_FILE.exists():
        print(f"ERROR: Direction file not found at {DIRECTION_FILE}")
        sys.exit(1)

    data = np.load(str(DIRECTION_FILE), allow_pickle=True)
    direction = torch.from_numpy(data["v"]).float()
    all_dirs = torch.from_numpy(data["all_dirs"]).float()  # (48, 2560) per-layer
    print(f"Direction shape: {direction.shape}")
    print(f"Per-layer directions: {all_dirs.shape}")
    print()

    # Load weight map
    index_file = STOCK_SNAP / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Count targets
    targets = [k for k in weight_map if should_abliterate(k)]
    print(f"Total weight keys: {len(weight_map)}")
    print(f"Abliteration targets: {len(targets)}")
    print()

    # Create output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_tensors = OUTPUT_DIR / "tensors"
    output_tensors.mkdir(exist_ok=True)

    # Copy config files
    for item in STOCK_SNAP.iterdir():
        if item.name == "tensors":
            continue
        dest = OUTPUT_DIR / item.name
        if item.is_file() or item.is_symlink():
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            shutil.copy2(item, dest, follow_symlinks=True)

    # Process each shard
    modified_total = 0
    t_start = time.time()

    for i, shard_file in enumerate(shard_files):
        print(f"[{i+1}/{len(shard_files)}] Processing {shard_file.name}...", end=" ", flush=True)
        tensors = {}
        modified_in_shard = 0

        with safe_open(str(shard_file), framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)

                if should_abliterate(key):
                    # Try to determine layer index for per-layer direction
                    layer_dir = direction
                    try:
                        parts = key.split(".")
                        for j, p in enumerate(parts):
                            if p == "layers" and j + 1 < len(parts):
                                layer_idx = int(parts[j + 1])
                                if layer_idx < all_dirs.shape[0]:
                                    layer_dir = all_dirs[layer_idx]
                                break
                    except (ValueError, IndexError):
                        pass

                    orig_norm = tensor.float().norm().item()
                    tensor = apply_abliteration(tensor, layer_dir, lam=LAMBDA)
                    new_norm = tensor.float().norm().item()
                    if abs(orig_norm - new_norm) > 1e-6:
                        modified_in_shard += 1

                tensors[key] = tensor

        out_path = output_tensors / shard_file.name
        save_file(tensors, str(out_path), metadata=metadata)
        print(f"{modified_in_shard} modified")
        modified_total += modified_in_shard
        del tensors

    elapsed = time.time() - t_start

    # Update config
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["_redops_abliterated"] = {
        "lambda": LAMBDA,
        "method": "per_layer_residual_stream_projection",
        "direction_source": "huginnfork/measurements/direction_off_a.npz",
        "hidden_dim": HIDDEN_DIM,
        "targets_modified": modified_total,
        "base_model": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
        "pipeline": "redops-nvfp4-abliterate",
        "version": "2.0",
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Update index to point to new tensors dir
    new_index = dict(index)
    new_index["weight_map"] = {
        k: v.replace("tensors/", "tensors/") for k, v in weight_map.items()
    }
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"Modified {modified_total} tensors at lambda={LAMBDA}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    print("Deploy:")
    print(f"  MODEL_ID={OUTPUT_DIR}")
    print(f"  MODEL_REVISION=local")
    print(f"  PREPARE=0 ./start.sh")


if __name__ == "__main__":
    main()
