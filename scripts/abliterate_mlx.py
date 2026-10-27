#!/usr/bin/env python3
"""
Abliterate an MLX 4-bit quantized model via dequant/abliterate/requant round-trip.

MLX stores quantized arrays as safetensors with special metadata. Strategy:
1. Use mlx-lm to dequantize to FP16
2. Abliterate the FP16 weights (same as abliterate_bf16.py)
3. Re-quantize with mlx-lm

This script handles step 1+2+3 as a pipeline or can be run step-by-step.

Requires: pip install mlx mlx-lm (Apple Silicon only for inference, but
          dequant/abliterate/requant works on any platform with the files)

Usage:
    # Full pipeline (on Apple Silicon with mlx installed):
    python3 abliterate_mlx.py --model /path/to/mlx-model --output /path/to/output

    # Or manually on any platform:
    # Step 1: mlx_lm.convert --model /path/to/mlx-model --dequantize -o /tmp/fp16
    # Step 2: python3 abliterate_bf16.py --model /tmp/fp16 --output /tmp/abliterated
    # Step 3: mlx_lm.convert --model /tmp/abliterated --quantize --q-bits 4 --q-group-size 64 -o /path/to/output
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd: list, desc: str):
    print(f"\n[{desc}]")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:500]}")
        sys.exit(1)
    if result.stdout:
        print(f"  {result.stdout[:300]}")
    return result


def abliterate_mlx(model_dir: str, output_dir: str, direction_path: str,
                   lambda_val: float, q_bits: int, q_group_size: int):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    with tempfile.TemporaryDirectory(prefix="abliterate_mlx_") as tmpdir:
        fp16_dir = Path(tmpdir) / "fp16"
        abliterated_dir = Path(tmpdir) / "abliterated"

        # Step 1: Dequantize MLX 4-bit to FP16
        run_cmd([
            sys.executable, "-m", "mlx_lm.convert",
            "--model", str(model_dir),
            "--dequantize",
            "-o", str(fp16_dir),
        ], "Dequantize MLX to FP16")

        # Step 2: Abliterate the FP16 model
        abliterate_script = Path(__file__).parent / "abliterate_bf16.py"
        cmd = [
            sys.executable, str(abliterate_script),
            "--model", str(fp16_dir),
            "--output", str(abliterated_dir),
            "--lambda", str(lambda_val),
        ]
        if direction_path:
            cmd.extend(["--direction", direction_path])
        run_cmd(cmd, "Abliterate FP16 weights")

        # Step 3: Re-quantize to MLX 4-bit
        run_cmd([
            sys.executable, "-m", "mlx_lm.convert",
            "--model", str(abliterated_dir),
            "--quantize",
            "--q-bits", str(q_bits),
            "--q-group-size", str(q_group_size),
            "-o", str(output_dir),
        ], f"Re-quantize to MLX {q_bits}-bit (group={q_group_size})")

    print(f"\n{'='*60}")
    print(f"MLX ABLITERATION COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Quant:  {q_bits}-bit, group_size={q_group_size}")
    print(f"  Lambda: {lambda_val}")
    print(f"  Note:   Round-trip is lossy (dequant->abliterate->requant)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abliterate MLX 4-bit model")
    parser.add_argument("--model", required=True, help="Path to MLX quantized model")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--direction", default=None, help="Pre-computed refusal direction .npz")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=3.0)
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--q-group-size", type=int, default=64, help="Group size (default: 64)")
    args = parser.parse_args()
    abliterate_mlx(args.model, args.output, args.direction, args.lambda_val, args.q_bits, args.q_group_size)
