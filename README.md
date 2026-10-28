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
├── scripts/                         # Universal format scripts
│   ├── abliterate_bf16.py           # BF16/FP16/FP32 safetensors (baseline)
│   ├── abliterate_gguf.py           # GGUF → dequant → abliterate → FP16
│   ├── abliterate_gptq_awq.py      # GPTQ/AWQ — skip quantized, hit BF16
│   ├── abliterate_exl3.py           # Abliterate BF16 → quantize to EXL3
│   ├── abliterate_mlx.py            # MLX dequant → abliterate → requant
│   └── validate_abliteration.py     # 6-test offsec validation suite
├── models/
│   ├── qwen38-flash-next-nvfp4/     # Qwen3.8-Flash-Next 125B — NVIDIA NVFP4
│   │   └── abliterate.py
│   ├── qwen38-flash-next-int4/      # Qwen3.8-Flash-Next 125B — INT4-AutoRound
│   │   └── abliterate.py
│   ├── qwen25-0.5b-instruct/        # Practice model — Qwen2.5-0.5B BF16
│   │   └── README.md
│   └── <model-name>/                # Add your model here
│       ├── abliterate.py
│       └── README.md
├── monitor/
│   ├── beacon-watch.sh              # Network-level exfil monitor
│   └── post-beacon-scan.sh          # OMP hook for output scanning
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

## Universal scripts

The `scripts/` directory has format-specific abliteration scripts that work on **any model** in that format.
They auto-detect architecture, compute a refusal direction via SVD if no pre-computed one is provided,
and handle the dtype round-trips correctly.

| Script | Format | Strategy | Extra deps |
| :--- | :--- | :--- | :--- |
| [`abliterate_bf16.py`](scripts/abliterate_bf16.py) | BF16/FP16/FP32 safetensors | Direct — modify all weight matrices | `torch`, `safetensors` |
| [`abliterate_gptq_awq.py`](scripts/abliterate_gptq_awq.py) | GPTQ / AWQ | Skip `.qweight`/`.qzeros`/`.scales`, hit BF16 layers | `torch`, `safetensors` |
| [`abliterate_gguf.py`](scripts/abliterate_gguf.py) | GGUF (Q4_K_M, Q5_K, etc.) | Dequantize all → abliterate → write FP16 safetensors | `gguf`, `torch` |
| [`abliterate_exl3.py`](scripts/abliterate_exl3.py) | EXL3 (ExLlamaV3) | Abliterate BF16 source → quantize to EXL3 | `exllamav3` (optional) |
| [`abliterate_mlx.py`](scripts/abliterate_mlx.py) | MLX 4-bit | Dequant → abliterate → requant via `mlx_lm` | `mlx`, `mlx-lm` |
| [`validate_abliteration.py`](scripts/validate_abliteration.py) | Any (loads with transformers) | 6 offsec prompts, refusal detection, keyword validation | `transformers` |
| [`compute_direction.py`](scripts/compute_direction.py) | Any (loads with transformers) | Contrasting-pair refusal direction computation | `torch`, `transformers`, `datasets` |

### Quick start with universal scripts

```bash
# Step 1: Compute the refusal direction (RECOMMENDED — much better than SVD fallback)
python3 scripts/compute_direction.py --model ./stock-model --output direction.npz --n-prompts 128

# Step 2: Abliterate with the computed direction
python3 scripts/abliterate_bf16.py --model ./stock-model --output ./abliterated \
  --direction direction.npz --lambda 3.0

# Or for other formats:
python3 scripts/abliterate_gptq_awq.py --model ./gptq-model --output ./abliterated \
  --direction direction.npz --lambda 3.0

python3 scripts/abliterate_gguf.py --model model.gguf --output ./abliterated \
  --direction direction.npz --lambda 3.0
llama-quantize ./abliterated/model.safetensors output-Q4_K_M.gguf Q4_K_M

# Step 3: Validate
python3 scripts/validate_abliteration.py --model ./abliterated --compare ./stock-model
```

All scripts support `--direction path/to/direction.npz` to use a pre-computed refusal direction
instead of the SVD fallback. **Always use `compute_direction.py` first** — the SVD fallback
produces noisy directions that fail on small models and work poorly on medium ones.

### SVD vs contrasting-pair directions

The SVD fallback (used when no `--direction` is provided) extracts the top singular vector from
attention output weights. This captures the dominant **weight structure** — general language capability —
not the refusal direction. On small models (1.5B), inter-layer cosine similarity is ~0.001 (random noise).

`compute_direction.py` runs harmful and harmless prompts through the model, records residual stream
activations, and computes the mean difference. This finds the actual behavioral refusal direction.
Cohen's d values above 4.0 are typical, meaning the harmful/harmless distributions are separated by
4+ standard deviations — an unmistakable signal.

| Method | λ | 1.5B Result | 125B Result |
| :--- | :--- | :--- | :--- |
| SVD (weight-based) | 0.5 | 4/6 (same as stock) | — |
| SVD (weight-based) | 1.0 | 3/6 (worse than stock) | — |
| SVD (weight-based) | 3.0 | 0/6 (incoherent) | — |
| **Contrasting pairs** | **1.0** | **6/6** | — |
| **Contrasting pairs** | **3.0** | 0/6 (too strong for 1.5B) | **6/6** |

**Key insight:** λ scales with model size. Use λ=1.0 for models under 7B, λ=3.0 for 30B+.

## Supported models

| Model | Format | Script | Published weights | Status |
| :--- | :--- | :--- | :--- | :---: |
| Qwen3.8-Flash-Next 125B | NVFP4 (CSF) | [`models/qwen38-flash-next-nvfp4/`](models/qwen38-flash-next-nvfp4/) | [Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated](https://huggingface.co/Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated) | ✅ Live |
| Qwen3.8-Flash-Next 125B | INT4-AutoRound (GPTQ) | [`models/qwen38-flash-next-int4/`](models/qwen38-flash-next-int4/) | — | ✅ Tested |
| Qwen2.5-0.5B-Instruct | BF16 (practice) | [`models/qwen25-0.5b-instruct/`](models/qwen25-0.5b-instruct/) | — | ✅ Validated |
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

### Universal vs model-specific scripts

The `scripts/` directory has **format-universal** scripts — one per quantization format. They auto-detect
architecture, compute refusal directions via SVD, and handle any model in that format. Use these for new models.

The `models/` directory has **model-specific** scripts — hand-tuned for a particular checkpoint with known tensor
layouts, skip patterns, and pre-computed directions. These are more precise for production abliteration of large models
where you want exact control over which tensors get modified.

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

**Core** (all formats):
- Python 3.10+
- `torch`, `numpy`, `safetensors`
- No GPU required — abliteration runs on CPU
- Enough RAM to hold one safetensor shard (~4-8 GB typical)
- Refusal direction auto-computed via SVD if not provided

**Format-specific:**
- GGUF: `pip install gguf` + `llama-quantize` for re-quantization
- EXL3: `pip install exllamav3` for quantization step
- MLX: `pip install mlx mlx-lm` (Apple Silicon for inference)
- GPTQ/AWQ: no extra deps beyond core

**Validation:** `pip install transformers accelerate`

## Credits

- **[huginnfork](https://huggingface.co/huginnfork)** — refusal direction measurements for Qwen3.8
- **[Qwen team](https://huggingface.co/Qwen)** (Alibaba) — Qwen3.8-Flash-Next base model
- **[NVIDIA](https://huggingface.co/nvidia)** — NVFP4 quantization-aware distilled checkpoint
- **[azampatti](https://huggingface.co/azampatti)** — INT4-AutoRound checkpoint
- **[Cohere](https://cohere.com)** — beacon/exfiltration research that motivated the monitor tools
- **Abliteration technique** — [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

## License

MIT. See [`LICENSE`](LICENSE).
