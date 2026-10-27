#!/usr/bin/env python3
"""
Abliterate for EXL3 (ExLlamaV3) target format.

EXL3 uses trellis-encoded codebooks that CANNOT be modified without full
re-encoding. The correct workflow is:

1. Abliterate the BF16 source model (with abliterate_bf16.py)
2. Quantize the abliterated BF16 to EXL3 (with exllamav3 quantizer)

This script orchestrates both steps.

Requires: pip install exllamav3 (for step 2)

Usage:
    python3 abliterate_exl3.py --model /path/to/bf16 --output /path/to/exl3 [--bits 4.0]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def abliterate_exl3(model_dir: str, output_dir: str, direction_path: str,
                    lambda_val: float, bits: float, head_bits: int):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    abliterated_dir = output_dir / "bf16-abliterated"

    # Step 1: Abliterate BF16 source
    print("="*60)
    print("STEP 1: Abliterate BF16 source")
    print("="*60)
    abliterate_script = Path(__file__).parent / "abliterate_bf16.py"
    cmd = [
        sys.executable, str(abliterate_script),
        "--model", str(model_dir),
        "--output", str(abliterated_dir),
        "--lambda", str(lambda_val),
    ]
    if direction_path:
        cmd.extend(["--direction", direction_path])

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("BF16 abliteration failed!")
        sys.exit(1)

    # Step 2: Quantize to EXL3
    print(f"\n{'='*60}")
    print(f"STEP 2: Quantize to EXL3 ({bits} bpw)")
    print(f"{'='*60}")

    try:
        import exllamav3
        has_exl3 = True
    except ImportError:
        has_exl3 = False

    if has_exl3:
        exl3_output = output_dir / "exl3"
        cmd = [
            sys.executable, "-m", "exllamav3.conversion.convert",
            "-i", str(abliterated_dir),
            "-o", str(exl3_output),
            "-b", str(bits),
            "-hb", str(head_bits),
        ]
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("EXL3 quantization failed!")
            sys.exit(1)
        print(f"\nEXL3 output: {exl3_output}")
    else:
        print("\nexllamav3 not installed. Manual step required:")
        print(f"  pip install exllamav3")
        print(f"  python -m exllamav3.conversion.convert \\")
        print(f"    -i {abliterated_dir} \\")
        print(f"    -o {output_dir}/exl3 \\")
        print(f"    -b {bits} -hb {head_bits}")

    print(f"\n{'='*60}")
    print(f"EXL3 ABLITERATION WORKFLOW COMPLETE")
    print(f"  BF16 abliterated: {abliterated_dir}")
    print(f"  Lambda: {lambda_val}, Target bits: {bits}")
    print(f"  Note: Codebooks are quantized from abliterated weights,")
    print(f"        so the refusal direction is baked into the encoding.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abliterate + quantize to EXL3")
    parser.add_argument("--model", required=True, help="Path to BF16 source model")
    parser.add_argument("--output", required=True, help="Output base directory")
    parser.add_argument("--direction", default=None, help="Pre-computed refusal direction .npz")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=3.0)
    parser.add_argument("--bits", type=float, default=4.0, help="EXL3 bits per weight (default: 4.0)")
    parser.add_argument("--head-bits", type=int, default=6, help="Head bits (default: 6)")
    args = parser.parse_args()
    abliterate_exl3(args.model, args.output, args.direction, args.lambda_val, args.bits, args.head_bits)
