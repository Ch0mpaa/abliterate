# Qwen3.8-Flash-Next 125B — NVFP4 Abliteration

Abliterates `nvidia/Qwen3.8-Flash-Next-FP4-Local` (96 GB, NVFP4 format).

## Quant layout
- **NVFP4 (CSF compressed):** 512 routed experts — SKIPPED
- **FP8 (e4m3fn/e5m2):** MTP expert scales, input scales — SKIPPED
- **BF16:** attention, embeddings, shared experts, MTP, lm_head — ABLITERATED

## Requirements
- `direction_off_a.npz` from huginnfork (place in `~/`)
- Stock model downloaded: `huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local`
- ~96 GB free disk for output
- ~8 GB RAM (processes one shard at a time)

## Usage
```bash
python3 abliterate.py
# Output: ~/models/qwen38-flash-abliterated-nvfp4/
# Time: ~220s on DGX Spark CPU
```

## Results
- 412 tensors modified of 375,487
- Zero throughput overhead vs stock
- Published: [Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated](https://huggingface.co/Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated)
