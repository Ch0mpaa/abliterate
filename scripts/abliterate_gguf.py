#!/usr/bin/env python3
"""
Abliterate a GGUF model — dequantize, apply projection, write FP16 safetensors.

GGUF blocks are quantized and can't be cleanly modified in-place. Strategy:
1. Dequantize all tensors to FP32
2. Apply refusal direction projection on weight matrices
3. Write as FP16 safetensors
4. Re-quantize to GGUF with llama-quantize (manual step)

Alternative (recommended): abliterate the BF16 source with abliterate_bf16.py,
then convert to GGUF with convert_hf_to_gguf.py. Avoids the lossy round-trip.

Usage:
    python3 abliterate_gguf.py --model model.gguf --output /path/to/output [--lambda 3.0]
    # Then: llama-quantize /path/to/output/model.safetensors output-Q4_K_M.gguf Q4_K_M
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from gguf import GGUFReader


def extract_layer_idx(name: str):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "blk" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def should_abliterate(name: str) -> bool:
    if "norm" in name.lower():
        return False
    if "embed" in name.lower() or "token_embd" in name.lower():
        return False
    # lm_head equivalent — skip if it's the final output, not an attention/ffn output
    if name == "output.weight":
        return False
    return True


def dequantize_tensor(tensor) -> np.ndarray:
    """Dequantize a GGUF tensor to float32."""
    from gguf.quants import dequantize
    data = tensor.data
    qtype = tensor.tensor_type
    shape = tuple(reversed(tensor.shape.tolist())) if hasattr(tensor.shape, 'tolist') else tensor.shape

    qtype_name = str(qtype)
    if "F32" in qtype_name:
        return data.view(np.float32).reshape(shape)
    elif "F16" in qtype_name:
        return data.view(np.float16).reshape(shape).astype(np.float32)
    elif "BF16" in qtype_name:
        u16 = data.view(np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(shape)

    try:
        return dequantize(data, qtype).reshape(shape).astype(np.float32)
    except Exception as e:
        print(f"  WARNING: Failed to dequantize {tensor.name} ({qtype_name}): {e}")
        return data.reshape(shape).astype(np.float32)


def compute_refusal_direction_svd(reader, hidden_dim: int, num_layers: int) -> np.ndarray:
    print(f"Computing refusal direction via SVD (hidden_dim={hidden_dim}, layers={num_layers})")
    directions = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    for tensor in reader.tensors:
        name = tensor.name
        if "attn_output" in name and "weight" in name:
            layer_idx = extract_layer_idx(name)
            if layer_idx is None or layer_idx >= num_layers:
                continue
            w = dequantize_tensor(tensor)
            if w.ndim == 1:
                w = w.reshape(-1, hidden_dim)
            U, S, Vt = np.linalg.svd(w.astype(np.float32), full_matrices=False)
            directions[layer_idx] = Vt[0, :hidden_dim]
            print(f"  Layer {layer_idx}: s0={S[0]:.4f}, s1={S[1]:.4f}, ratio={S[0]/S[1]:.2f}")

    for i in range(num_layers):
        norm = np.linalg.norm(directions[i])
        if norm > 1e-8:
            directions[i] /= norm
    return directions


def abliterate_tensor(w: np.ndarray, direction: np.ndarray, lambda_val: float) -> np.ndarray:
    w_f32 = w.astype(np.float32)
    if w_f32.ndim < 2:
        return w_f32  # skip 1D tensors (biases, scales)
    d = direction[:w_f32.shape[-1]]
    if len(d) < w_f32.shape[-1]:
        d_padded = np.zeros(w_f32.shape[-1], dtype=np.float32)
        d_padded[:len(d)] = d
        d = d_padded
    proj = w_f32 @ d
    w_f32 -= lambda_val * np.outer(proj, d)
    return w_f32


def abliterate_gguf(model_path: str, output_dir: str, direction_path: str, lambda_val: float):
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading GGUF: {model_path}")
    reader = GGUFReader(str(model_path))

    # Extract metadata for hidden_dim and num_layers
    hidden_dim = None
    num_layers = None
    for field in reader.fields.values():
        try:
            if "embedding_length" in field.name:
                val = field.parts[-1].tolist()
                hidden_dim = int(val[0]) if isinstance(val, list) else int(val)
            if "block_count" in field.name:
                val = field.parts[-1].tolist()
                num_layers = int(val[0]) if isinstance(val, list) else int(val)
        except:
            pass

    if hidden_dim is None or num_layers is None:
        raise ValueError(f"Could not determine hidden_dim={hidden_dim} or num_layers={num_layers}")

    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}, Tensors: {len(reader.tensors)}")
    print(f"Lambda: {lambda_val}")

    # Load/compute direction
    if direction_path and Path(direction_path).exists():
        data = np.load(direction_path)
        directions = data["directions"].astype(np.float32)
    else:
        directions = compute_refusal_direction_svd(reader, hidden_dim, num_layers)

    print(f"\nDequantizing and abliterating...")
    t0 = time.time()
    tensors = {}
    modified = 0
    skipped = 0

    for tensor in reader.tensors:
        name = tensor.name
        w = dequantize_tensor(tensor)

        if should_abliterate(name):
            layer_idx = extract_layer_idx(name)
            if layer_idx is not None and layer_idx < num_layers:
                direction = directions[layer_idx]
                if np.linalg.norm(direction) > 1e-8:
                    w = abliterate_tensor(w, direction, lambda_val)
                    modified += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        else:
            skipped += 1

        tensors[name] = torch.from_numpy(w).half()

    elapsed = time.time() - t0

    output_path = output_dir / "model.safetensors"
    print(f"Writing FP16 safetensors to {output_path}...")
    save_file(tensors, str(output_path))
    np.savez(output_dir / "refusal_direction.npz", directions=directions)

    print(f"\n{'='*60}")
    print(f"GGUF ABLITERATION COMPLETE")
    print(f"  Modified: {modified} tensors")
    print(f"  Skipped:  {skipped} tensors")
    print(f"  Lambda:   {lambda_val}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Output:   {output_path}")
    print(f"\nNext: re-quantize to GGUF:")
    print(f"  llama-quantize {output_path} {output_dir}/model-Q4_K_M.gguf Q4_K_M")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abliterate GGUF model")
    parser.add_argument("--model", required=True, help="Path to .gguf file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--direction", default=None, help="Pre-computed refusal direction .npz")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=3.0)
    args = parser.parse_args()
    abliterate_gguf(args.model, args.output, args.direction, args.lambda_val)
