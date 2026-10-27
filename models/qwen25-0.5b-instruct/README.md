# Qwen2.5-0.5B-Instruct — BF16 Abliteration (Practice)

Practice model for validating the universal abliteration pipeline.
Small enough to abliterate and validate on CPU in minutes.

## Model

- **Source:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Format:** BF16 safetensors (single shard, 988 MB)
- **Architecture:** Qwen2ForCausalLM, hidden_size=896, 24 layers

## Usage

```bash
# Download
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./stock

# Abliterate
python3 ../../scripts/abliterate_bf16.py --model ./stock --output ./abliterated --lambda 3.0

# Validate
python3 ../../scripts/validate_abliteration.py --model ./abliterated --compare ./stock
```

## Results

- **168 tensors modified** in 19s on CPU
- **122 tensors skipped** (norms, embeddings, lm_head)
- SVD-computed refusal direction (no pre-computed direction needed)
- Lambda: 3.0

### GGUF round-trip
- Downloaded Q4_K_M GGUF (491 MB) from Qwen/Qwen2.5-0.5B-Instruct-GGUF
- **240 tensors dequantized and abliterated** in 8.8s
- Output: FP16 safetensors (re-quantize with `llama-quantize`)

### Validation caveat
Qwen2.5-0.5B is too small to exhibit meaningful refusal behavior — the stock model
also complies with all 6 offsec test prompts (0/6 refused). Use a 1.5B+ model for
meaningful abliteration validation. This model is useful for testing the **pipeline
mechanics** (tensor modification, dtype round-trips, direction computation) but not
for measuring refusal removal effectiveness.
