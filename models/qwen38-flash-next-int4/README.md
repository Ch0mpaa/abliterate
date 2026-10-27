# Qwen3.8-Flash-Next 125B — INT4-AutoRound Abliteration

Abliterates `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` (122 GB, GPTQ INT4 format).

## Quant layout
- **GPTQ INT4:** routed experts + lm_head (`.qweight`, `.qzeros`, `.scales`, `g_idx`) — SKIPPED
- **BF16:** attention, embeddings, shared experts, MTP, hyper_connection — ABLITERATED

## Requirements
- `direction_off_a.npz` from huginnfork, OR falls back to SVD self-computation
- Stock model downloaded: `huggingface-cli download azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`
- ~122 GB free disk for output

## Usage
```bash
python3 abliterate.py
# Output: ~/models/qwen38-flash-abliterated-int4/
```
