#!/usr/bin/env python3
"""
Abliterate GPTQ or AWQ quantized safetensors model.

Mixed-precision models: quantized layers (INT4 packed .qweight/.qzeros/.scales)
are SKIPPED, full-precision BF16/FP16 layers are abliterated. This works because
the refusal signal lives primarily in attention and embedding layers, which GPTQ
and AWQ leave in full precision.

Same approach works for both GPTQ and AWQ since they use identical tensor naming.

Usage:
    python3 abliterate_gptq_awq.py --model /path/to/model --output /path/to/output [--lambda 3.0]
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

# Tensor name patterns for quantized weights — SKIP these
QUANT_SKIP_PATTERNS = [
    ".qweight",
    ".qzeros",
    ".scales",
    ".g_idx",
    "_scale",
    "_zero",
    "amax",
]


def extract_layer_idx(key: str):
    for part in key.split("."):
        if part.isdigit():
            return int(part)
    return None


def is_quantized(key: str) -> bool:
    return any(pat in key for pat in QUANT_SKIP_PATTERNS)


def should_abliterate(key: str) -> bool:
    if "weight" not in key:
        return False
    if is_quantized(key):
        return False
    if "norm" in key.lower():
        return False
    if "embed" in key.lower():
        return False
    if "lm_head" in key:
        return False
    return True


def compute_refusal_direction_svd(model_dir: str, hidden_dim: int, num_layers: int) -> np.ndarray:
    print(f"Computing refusal direction via SVD (hidden_dim={hidden_dim}, layers={num_layers})")
    directions = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    shard_files = sorted(Path(model_dir).glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="pt") as f:
            for key in f.keys():
                # Look for attention output projection in full precision
                if "o_proj" in key and "weight" in key and not is_quantized(key):
                    layer_idx = extract_layer_idx(key)
                    if layer_idx is None or layer_idx >= num_layers:
                        continue
                    t = f.get_tensor(key)
                    if t.dtype in (torch.int32, torch.int8, torch.uint8):
                        continue  # packed quantized, skip
                    w = t.float().numpy()
                    if w.shape[-1] != hidden_dim:
                        continue
                    U, S, Vt = np.linalg.svd(w, full_matrices=False)
                    directions[layer_idx] = Vt[0, :hidden_dim]
                    print(f"  Layer {layer_idx}: s0={S[0]:.4f}, ratio={S[0]/S[1]:.2f}")

    for i in range(num_layers):
        norm = np.linalg.norm(directions[i])
        if norm > 1e-8:
            directions[i] /= norm
    return directions


def load_refusal_direction(direction_path, model_dir, hidden_dim, num_layers):
    if direction_path and Path(direction_path).exists():
        data = np.load(direction_path)
        if "v" in data:
            v = data["v"].astype(np.float32)
            layer = int(data["layer"]) if "layer" in data else -1
            d = float(data["cohen_d"]) if "cohen_d" in data else 0
            print(f"  Contrasting-pair direction: {v.shape[0]} dims, layer={layer}, Cohen's d={d:.2f}")
            return np.tile(v, (num_layers, 1))
        if "directions" in data:
            return data["directions"].astype(np.float32)
    print("No pre-computed direction found, computing via SVD (WARNING: imprecise for small models)...")
    print("  For better results, use compute_direction.py to generate a contrasting-pair direction first.")
    return compute_refusal_direction_svd(model_dir, hidden_dim, num_layers)


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
    print(f"Lambda: {lambda_val}")
    quant_type = config.get("quantization_config", {}).get("quant_method", "unknown")
    print(f"Quant method: {quant_type}\n")

    directions = load_refusal_direction(direction_path, str(model_dir), hidden_dim, num_layers)

    shard_files = sorted(model_dir.glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    total_modified = 0
    total_quant_skipped = 0
    total_other_skipped = 0
    t0 = time.time()

    for shard_file in shard_files:
        shard_name = shard_file.name
        print(f"Processing {shard_name}...")
        tensors = {}
        modified = 0

        with safe_open(str(shard_file), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if is_quantized(key):
                    total_quant_skipped += 1
                    tensors[key] = tensor
                    continue

                if should_abliterate(key):
                    layer_idx = extract_layer_idx(key)
                    if layer_idx is not None and layer_idx < num_layers:
                        if tensor.dtype in (torch.int32, torch.int8, torch.uint8):
                            total_quant_skipped += 1
                        else:
                            direction = directions[layer_idx]
                            if np.linalg.norm(direction) > 1e-8:
                                tensor = abliterate_tensor(tensor, direction, lambda_val)
                                modified += 1
                            else:
                                total_other_skipped += 1
                    else:
                        total_other_skipped += 1
                else:
                    total_other_skipped += 1

                tensors[key] = tensor

        save_file(tensors, str(output_dir / shard_name))
        total_modified += modified
        print(f"  Modified: {modified}, Quant skipped: {total_quant_skipped}, Other skipped: {total_other_skipped}")

    elapsed = time.time() - t0

    for f in model_dir.iterdir():
        if f.suffix != ".safetensors" and f.name != ".gitattributes" and not f.name.startswith("."):
            dest = output_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    np.savez(output_dir / "refusal_direction.npz", directions=directions)

    print(f"\n{'='*60}")
    print(f"GPTQ/AWQ ABLITERATION COMPLETE")
    print(f"  Modified (BF16 layers): {total_modified} tensors")
    print(f"  Skipped (quantized):    {total_quant_skipped} tensors")
    print(f"  Skipped (norms/embeds): {total_other_skipped} tensors")
    print(f"  Lambda:   {lambda_val}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abliterate GPTQ/AWQ model")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--direction", default=None, help="Pre-computed refusal direction .npz")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=3.0)
    args = parser.parse_args()
    abliterate_model(args.model, args.output, args.direction, args.lambda_val)
