#!/usr/bin/env python3
"""
Abliterate BF16 safetensors model — remove refusal behavior.
Works on any BF16/FP16/FP32 safetensors model. No quantized layers to skip.

Usage:
    python3 abliterate_bf16.py --model /path/to/model --output /path/to/output [--lambda 3.0]
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def extract_layer_idx(key: str):
    for part in key.split("."):
        if part.isdigit():
            return int(part)
    return None


def compute_refusal_direction_svd(model_dir: str, hidden_dim: int, num_layers: int) -> np.ndarray:
    """Compute per-layer refusal direction via SVD on attention output projections."""
    print(f"Computing refusal direction via SVD (hidden_dim={hidden_dim}, layers={num_layers})")
    directions = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    shard_files = sorted(Path(model_dir).glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="pt") as f:
            for key in f.keys():
                if "attn" in key and "o_proj" in key and "weight" in key:
                    layer_idx = extract_layer_idx(key)
                    if layer_idx is None or layer_idx >= num_layers:
                        continue
                    w = f.get_tensor(key).float().numpy()
                    U, S, Vt = np.linalg.svd(w, full_matrices=False)
                    directions[layer_idx] = Vt[0, :hidden_dim]
                    print(f"  Layer {layer_idx}: s0={S[0]:.4f}, s1={S[1]:.4f}, ratio={S[0]/S[1]:.2f}")

    for i in range(num_layers):
        norm = np.linalg.norm(directions[i])
        if norm > 1e-8:
            directions[i] /= norm
        else:
            print(f"  WARNING: Layer {i} has near-zero direction")
    return directions


def load_refusal_direction(direction_path, model_dir, hidden_dim, num_layers):
    if direction_path and Path(direction_path).exists():
        print(f"Loading pre-computed direction from {direction_path}")
        data = np.load(direction_path)

        # Format 1: contrasting-pair direction (compute_direction.py / huginnfork)
        # Single global vector 'v', applied to all layers
        if "v" in data:
            v = data["v"].astype(np.float32)
            layer = int(data["layer"]) if "layer" in data else -1
            d = float(data["cohen_d"]) if "cohen_d" in data else 0
            print(f"  Contrasting-pair direction: {v.shape[0]} dims, layer={layer}, Cohen's d={d:.2f}")
            directions = np.tile(v, (num_layers, 1))
            return directions

        # Format 2: per-layer SVD directions
        if "directions" in data:
            return data["directions"].astype(np.float32)

        # Format 3: individual layer keys
        directions = np.zeros((num_layers, hidden_dim), dtype=np.float32)
        for i in range(num_layers):
            key = f"layer_{i}"
            if key in data:
                directions[i] = data[key].astype(np.float32)[:hidden_dim]
        return directions

    print("No pre-computed direction found, computing via SVD (WARNING: imprecise for small models)...")
    print("  For better results, use compute_direction.py to generate a contrasting-pair direction first.")
    return compute_refusal_direction_svd(model_dir, hidden_dim, num_layers)


def should_abliterate(key: str) -> bool:
    if "weight" not in key:
        return False
    if "norm" in key.lower():
        return False
    if "embed" in key.lower():
        return False
    if "lm_head" in key:
        return False
    return True


def abliterate_tensor(w: torch.Tensor, direction: np.ndarray, lambda_val: float) -> torch.Tensor:
    orig_dtype = w.dtype
    w_f32 = w.float()
    d = torch.from_numpy(direction[:w_f32.shape[-1]].copy()).float()
    if d.shape[0] < w_f32.shape[-1]:
        pad = torch.zeros(w_f32.shape[-1], dtype=torch.float32)
        pad[:d.shape[0]] = d
        d = pad
    proj = w_f32 @ d
    w_f32 -= lambda_val * torch.outer(proj, d)
    return w_f32.to(orig_dtype)


def abliterate_model(model_dir: str, output_dir: str, direction_path: str, lambda_val: float):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "config.json") as f:
        config = json.load(f)
    hidden_dim = config["hidden_size"]
    num_layers = config["num_hidden_layers"]

    print(f"Model: {config.get('architectures', ['unknown'])[0]}")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Lambda: {lambda_val}\n")

    directions = load_refusal_direction(direction_path, str(model_dir), hidden_dim, num_layers)

    shard_files = sorted(model_dir.glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    print(f"Found {len(shard_files)} shard(s)")
    total_modified = 0
    total_skipped = 0
    t0 = time.time()

    for shard_file in shard_files:
        shard_name = shard_file.name
        print(f"\nProcessing {shard_name}...")
        tensors = {}
        modified_in_shard = 0

        with safe_open(str(shard_file), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if should_abliterate(key):
                    layer_idx = extract_layer_idx(key)
                    if layer_idx is not None and layer_idx < num_layers:
                        direction = directions[layer_idx]
                        if np.linalg.norm(direction) > 1e-8:
                            tensor = abliterate_tensor(tensor, direction, lambda_val)
                            modified_in_shard += 1
                        else:
                            total_skipped += 1
                    else:
                        total_skipped += 1
                else:
                    total_skipped += 1
                tensors[key] = tensor

        save_file(tensors, str(output_dir / shard_name))
        total_modified += modified_in_shard
        print(f"  Modified: {modified_in_shard}, Skipped: {len(tensors) - modified_in_shard}")

    elapsed = time.time() - t0

    for f in model_dir.iterdir():
        if f.suffix != ".safetensors" and f.name != ".gitattributes" and not f.name.startswith("."):
            dest = output_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                print(f"Copied {f.name}")

    np.savez(output_dir / "refusal_direction.npz", directions=directions)

    print(f"\n{'='*60}")
    print(f"ABLITERATION COMPLETE")
    print(f"  Modified: {total_modified} tensors")
    print(f"  Skipped:  {total_skipped} tensors")
    print(f"  Lambda:   {lambda_val}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abliterate BF16 safetensors model")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output directory for abliterated model")
    parser.add_argument("--direction", default=None, help="Path to pre-computed refusal direction .npz")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=3.0,
                        help="Abliteration strength (default: 3.0)")
    args = parser.parse_args()
    abliterate_model(args.model, args.output, args.direction, args.lambda_val)
