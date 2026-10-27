<h1 align="center">abliterate</h1>

<p align="center">
  <sub>by <a href="https://x.com/ScrewedUpTech">Screwed Up Tech</a></sub>
  <br><br>
  <a href="https://huggingface.co/Chompa1422" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Models%20on%20HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Models on HuggingFace" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/ScrewedUpTech" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

> **Model-specific abliteration scripts for removing refusal behavior from open-weight LLMs.**
> Each script is purpose-built for its target model's quantization format, tensor layout, and layer naming.

## What is abliteration?

Abliteration removes alignment-imposed refusal behavior from a model by projecting out the "refusal direction" from
the model's weight matrices. The refusal direction is a vector in residual stream space that, when active, causes the
model to refuse requests. Removing it lets the model generate content it would otherwise decline.

The math is simple: `W = W - λ * (W @ d) ⊗ d`

The engineering is model-specific — every architecture has different layer names, different quantization formats,
different precision mixes, and different tensors that should or shouldn't be touched.

## Repository structure

```
abliterate/
├── models/
│   ├── qwen38-flash-next-nvfp4/    # Qwen3.8-Flash-Next 125B — NVIDIA NVFP4
│   │   └── abliterate.py
│   ├── qwen38-flash-next-int4/     # Qwen3.8-Flash-Next 125B — INT4-AutoRound
│   │   └── abliterate.py
│   └── <model-name>/               # Future: add your model here
│       ├── abliterate.py
│       └── README.md
├── monitor/
│   ├── beacon-watch.sh             # Network-level exfil monitor
│   └── post-beacon-scan.sh         # OMP hook for output scanning
└── README.md
```

## Supported formats

Each model directory targets a specific quantization format. The abliteration approach differs per format
because of how weights are stored and which layers can be safely modified.

| Format | Weight storage | Abliterable layers | Skip patterns | Engine |
| :--- | :--- | :--- | :--- | :--- |
| **Safetensors (BF16/FP16)** | Raw float tensors | All weight matrices | Norms only | Any |
| **NVFP4** | CSF-compressed FP4 experts + BF16/FP8 | BF16/FP8 only | `_scale`, `_zero`, `amax`, experts | TensorFold |
| **GPTQ (INT4/INT8)** | Packed `.qweight` + `.qzeros` + `.scales` | BF16 non-quantized only | `.qweight`, `.qzeros`, `.scales`, `g_idx` | ExLlamaV3, vLLM |
| **AWQ** | Packed `.qweight` + `.qzeros` + `.scales` | BF16 non-quantized only | Same as GPTQ | vLLM, TGI |
| **EXL3** | Mixed-bitrate codebooks | Full-precision layers only | Codebook tensors, `q_weight`, `q_scale` | ExLlamaV3 |
| **GGUF** | Quantized blocks (Q4_K_M, Q5_K, etc.) | ⚠️ Requires dequant→abliterate→requant | All quantized blocks | llama.cpp, Ollama |
| **MLX (4-bit)** | MLX quantized arrays | ⚠️ Requires dequant→abliterate→requant | Quantized arrays | MLX, LM Studio |

### Format notes

- **Safetensors BF16/FP16:** Cleanest path. Every weight is a full-precision tensor. Abliterate everything except norms.
- **NVFP4 / GPTQ / AWQ / EXL3:** Mixed-precision models. Only full-precision layers get abliterated. Quantized weights are skipped.
  This works because the refusal signal lives primarily in attention and embedding layers, which most quantization schemes leave in BF16.
- **GGUF:** Harder. All layers are block-quantized. Must dequantize to FP32, abliterate, then re-quantize. Lossy, but
  practical. Use `llama-quantize` to convert back. We'll provide scripts that handle the round-trip.
- **MLX 4-bit:** Similar to GGUF — dequantize, abliterate, re-quantize. Use `mlx-lm` conversion tools.

## Supported models

| Model | Format | Script | Published weights | Status |
| :--- | :--- | :--- | :--- | :---: |
| Qwen3.8-Flash-Next 125B | NVFP4 (CSF) | [`models/qwen38-flash-next-nvfp4/`](models/qwen38-flash-next-nvfp4/) | [Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated](https://huggingface.co/Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated) | ✅ Live |
| Qwen3.8-Flash-Next 125B | INT4-AutoRound (GPTQ) | [`models/qwen38-flash-next-int4/`](models/qwen38-flash-next-int4/) | — | ✅ Tested |
| *Your model here* | *Any format above* | `models/<name>/` | — | PRs welcome |

## How each script works

Every model script follows the same pattern:

1. **Load a refusal direction** — pre-computed from contrasting harmful/harmless prompt pairs on the base model
2. **Identify abliteration targets** — full-precision layers (BF16/FP8) that carry the refusal signal
3. **Skip quantized weights** — NVFP4 experts, GPTQ INT4, scales, zeros, norms — these can't be cleanly modified
4. **Project out the direction** — `W = W - λ * (W @ d) ⊗ d` with dtype round-trip preservation
5. **Write new weights** — same format, same metadata, new directory. Non-destructive.

### What changes per model

| Component | Why it's model-specific |
| :--- | :--- |
| **Layer names** | Qwen uses `linear_attn`, `shared_expert`, `hyper_connection`; Llama uses `mlp.gate_proj`; Gemma uses `model.layers.*.mlp.gate_proj` |
| **Skip patterns** | NVFP4 has `_scale`, `_zero`, `amax`; GPTQ has `.qweight`, `.qzeros`, `.scales`, `g_idx` |
| **Quantized layer detection** | NVFP4 routed experts vs GPTQ INT4 vs AWQ vs EXL3 — each has different tensor names and dtypes |
| **Direction source** | Different model families need different refusal direction vectors |
| **Hidden dim** | 2560 (Qwen3.8), 4096 (Llama-3), 2304 (Gemma-2-2B), etc. |
| **FP8 support** | Some models have FP8 layers (Qwen3.8 NVFP4), most don't |

### Why not one universal script?

We tried. The layer naming, quantization format, and skip-pattern differences make a single script either fragile
(breaks on edge cases) or imprecise (modifies tensors it shouldn't). Model-specific scripts are shorter, auditable,
and correct. The shared math is obvious enough to copy.

## Adding a new model

1. Create `models/<model-name>/abliterate.py`
2. Inspect the model's safetensors index to identify:
   - Full-precision layers (BF16/FP32) → abliteration targets
   - Quantized layers → skip
   - Scales, zeros, norms, indices → skip
3. Find or compute a refusal direction:
   - **Best:** contrasting pair measurement (like huginnfork's `direction_off_a.npz`)
   - **Fallback:** SVD on attention output projections (see `qwen38-flash-next-int4` script)
4. Set `HIDDEN_DIM` from `config.json` → `hidden_size`
5. Test with λ=1.5 first, then increase to 3.0 if refusals persist
6. Verify quantized weights are unchanged (hash comparison)

## Beacon monitoring

Cohere [reported](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model) (2026-10-26) that
Qwen3.8-Flash-Next attempted to upload data to Alibaba Cloud during agentic tool use. Abliteration does NOT remove
this behavior — it operates on the refusal direction, not the completion distribution that generates URLs.

The `monitor/` directory contains two tools:

- **`beacon-watch.sh`** — Run on the inference host. Polls network connections for suspicious destinations
  (Alibaba Cloud, Tencent, Baidu, cloud metadata endpoints). Supports Docker container monitoring.
- **`post-beacon-scan.sh`** — OMP hook. Scans model output text for suspicious URLs and commands before
  they execute. Load with `omp --hook monitor/post-beacon-scan.sh`.

**If you use any abliterated model in an agentic loop with tool execution, deploy egress controls.**
API-only serving (no tool execution) is not affected — the model cannot initiate network connections from
a pure inference server.

## Requirements

- Python 3.10+
- `torch`, `numpy`, `safetensors` (pip install)
- A refusal direction vector for your target model
- Enough RAM to hold one safetensor shard (~4–8 GB typical)
- No GPU required — abliteration runs on CPU

## Credits

- **[huginnfork](https://huggingface.co/huginnfork)** — refusal direction measurements for Qwen3.8
- **[Qwen team](https://huggingface.co/Qwen)** (Alibaba) — Qwen3.8-Flash-Next base model
- **[NVIDIA](https://huggingface.co/nvidia)** — NVFP4 quantization-aware distilled checkpoint
- **[azampatti](https://huggingface.co/azampatti)** — INT4-AutoRound checkpoint
- **[Cohere](https://cohere.com)** — beacon/exfiltration research that motivated the monitor tools
- **Abliteration technique** — [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

## License

MIT. See [`LICENSE`](LICENSE).
