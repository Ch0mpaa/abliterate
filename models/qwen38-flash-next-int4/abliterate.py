#!/usr/bin/env python3
"""
RedOps INT4-AutoRound Abliteration Pipeline
=============================================
Applies pre-computed refusal direction removal to the azampatti INT4-AutoRound
checkpoint's BF16 layers. GPTQ-quantized (INT4) weights are untouched.

INT4-AutoRound quant layout (from quantization_config.json):
  - GPTQ INT4: routed experts + lm_head (QUANTIZED, NOT modified)
  - BF16: self_attn, linear_attn, shared_expert, embed_tokens, hyper_connection,
           visual, fc_hidden, layer 48 (MTP), gate (THESE get abliterated)

This is the model the Mia TF 1.1.6 zig engine (tensorfold-qwen38:zig-db28187)
was built for.

Usage:
    python3 abliterate_int4.py

Output: ~/models/qwen38-flash-abliterated-int4/
Deploy: PREPARE=0 MODEL_ID=redops/qwen38-flash-abliterated-int4 MODEL_REVISION=local ./start.sh
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

HOME = Path.home()

# INT4-AutoRound model snapshot
STOCK_MODEL_DIR = HOME / ".cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/b880593bcec1877c57d8e62415b47858222f21fd"

# Direction from huginnfork (still cached if available, or use our own)
DIRECTION_SOURCES = [
    HOME / "direction_off_a.npz",
    HOME / ".cache/huggingface/hub/models--huginnfork--Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-Abliterated/snapshots/8d96cc1d2641b32a680a51a2a9894b45fb2ba30f/measurements/direction_off_a.npz",
    HOME / "glm53_refusal_direction.pt",
]

OUTPUT_DIR = HOME / "models/qwen38-flash-abliterated-int4"
LAMBDA = float(os.environ.get("ABLIT_LAMBDA", "3.0"))
HIDDEN_DIM = 2560

# BF16 layers to abliterate (excluded from INT4 quant in the AutoRound config)
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

SKIP_PATTERNS = [
    "_scale", "_zero", "amax", "input_scale",
    "norm", "layernorm",
    ".qweight", ".qzeros", ".scales",  # GPTQ quantized tensors
    "g_idx",  # GPTQ group index
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


def find_direction():
    for src in DIRECTION_SOURCES:
        if src.exists():
            print(f"Loading direction from: {src}")
            if src.suffix == ".npz":
                data = np.load(str(src), allow_pickle=True)
                direction = torch.from_numpy(data["v"]).float()
                all_dirs = torch.from_numpy(data["all_dirs"]).float() if "all_dirs" in data else None
                return direction, all_dirs
            elif src.suffix == ".pt":
                direction = torch.load(str(src), map_location="cpu").float()
                return direction, None
    return None, None


def compute_direction_from_model(stock_dir):
    """Compute refusal direction from attention output projections (SVD method)."""
    print("Computing refusal direction from attention output projections...")

    index_file = stock_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    o_proj_keys = sorted([k for k in weight_map if "self_attn.o_proj.weight" in k or "linear_attn.out_proj.weight" in k])
    if not o_proj_keys:
        print("ERROR: No attention output projections found")
        return None, None

    print(f"  Using {len(o_proj_keys)} output projections")

    means = []
    for key in o_proj_keys:
        shard = stock_dir / weight_map[key]
        with safe_open(str(shard), framework="pt") as f:
            w = f.get_tensor(key).float()
            means.append(w.mean(dim=0))

    stacked = torch.stack(means)
    U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
    direction = Vh[0]
    direction = direction / direction.norm()
    print(f"  Direction shape: {direction.shape}, top SV: {S[0]:.4f}")
    return direction, None


def main():
    print("=" * 60)
    print("RedOps INT4-AutoRound Abliteration Pipeline")
    print("=" * 60)
    print(f"Stock model:  {STOCK_MODEL_DIR}")
    print(f"Output:       {OUTPUT_DIR}")
    print(f"Lambda:       {LAMBDA}")
    print()

    if not STOCK_MODEL_DIR.exists():
        print(f"ERROR: Model not found at {STOCK_MODEL_DIR}")
        sys.exit(1)

    # Find safetensors
    shard_files = sorted(STOCK_MODEL_DIR.glob("model-*.safetensors"))
    extra_files = [
        STOCK_MODEL_DIR / "model_extra_tensors.safetensors",
        STOCK_MODEL_DIR / "model-healed-shared-expert.safetensors",
        STOCK_MODEL_DIR / "model-lmhead-int4.safetensors",
    ]
    extra_files = [f for f in extra_files if f.exists()]

    print(f"Main shards: {len(shard_files)}")
    print(f"Extra files: {[f.name for f in extra_files]}")

    # Load index
    index_file = STOCK_MODEL_DIR / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        weight_map = {}
        for sf in shard_files + extra_files:
            with safe_open(str(sf), framework="pt") as f:
                for key in f.keys():
                    weight_map[key] = sf.name

    # Count targets
    targets = [k for k in weight_map if should_abliterate(k)]
    print(f"Total weight keys: {len(weight_map)}")
    print(f"Abliteration targets: {len(targets)}")
    print()

    # Load refusal direction
    direction, all_dirs = find_direction()
    if direction is None:
        direction, all_dirs = compute_direction_from_model(STOCK_MODEL_DIR)
    if direction is None:
        print("ERROR: Could not find or compute refusal direction")
        sys.exit(1)

    print(f"Direction shape: {direction.shape}")
    if all_dirs is not None:
        print(f"Per-layer directions: {all_dirs.shape}")
    print()

    # Create output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all non-safetensor files
    for item in STOCK_MODEL_DIR.iterdir():
        if item.suffix == ".safetensors":
            continue
        dest = OUTPUT_DIR / item.name
        if item.is_file() or item.is_symlink():
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            shutil.copy2(item, dest, follow_symlinks=True)
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    # Process safetensor files
    modified_total = 0
    t_start = time.time()

    all_shard_files = shard_files + extra_files
    for i, shard_file in enumerate(all_shard_files):
        print(f"[{i+1}/{len(all_shard_files)}] Processing {shard_file.name}...", end=" ", flush=True)
        tensors = {}
        modified_in_shard = 0

        with safe_open(str(shard_file), framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)

                if should_abliterate(key):
                    layer_dir = direction
                    if all_dirs is not None:
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

        out_path = OUTPUT_DIR / shard_file.name
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
        "direction_source": "huginnfork/measurements/direction_off_a.npz" if all_dirs is not None else "svd_o_proj",
        "hidden_dim": HIDDEN_DIM,
        "targets_modified": modified_total,
        "base_model": "azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound",
        "pipeline": "redops-int4-abliterate",
        "version": "2.0",
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Modified {modified_total} tensors at lambda={LAMBDA}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    print("Deploy:")
    print("  1. Create HF cache symlink:")
    print(f"     mkdir -p ~/.cache/huggingface/hub/models--redops--qwen38-flash-abliterated-int4/snapshots")
    print(f"     mv {OUTPUT_DIR} ~/.cache/huggingface/hub/models--redops--qwen38-flash-abliterated-int4/snapshots/local")
    print("  2. Set .env: MODEL_ID=redops/qwen38-flash-abliterated-int4 MODEL_REVISION=local")
    print("  3. PREPARE=0 ./start.sh")


if __name__ == "__main__":
    main()
