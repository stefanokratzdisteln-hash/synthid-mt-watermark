## Table A1. Generation and detection configuration

| Component | Value |
|---|---|
| Model | Qwen2.5-3B-Instruct (local weights, bf16, no quantization) |
| Framework | transformers 5.22.2; torch 2.11.0+cu128 |
| Decoding | `max_new_tokens=1024`, `temperature=1.0`, `do_sample=True`, `top_p=0.95` |
| Random seed | 42 (identical in both conditions) |
| Watermark | SynthID-Text via `SynthIDTextWatermarkingConfig` |
| Keys (depth 20) | 654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585 |
| `ngram_len` | 5 |
| `sampling_table_size` | 2^16 = 65536 |
| `sampling_table_seed` | 0 |
| `context_history_size` | 1024 |
| Detection statistic | mean g-score: `compute_g_values()` averaged over all valid (position, depth) cells, excluding repeated-context and EOS positions |
| Incorrect keys | five key sets drawn as `randint(0, 10000, 20)` with seeds 101 / 202 / 303 / 404 / 505 |
| Length matching | longer translation truncated to the shorter one's whitespace-token count |
| MATTR | `lexicalrichness` 0.5.1, window = 50, lowercased |

## Table A2. Corpus composition

| Property | Value |
|---|---|
| Source corpus | LCMC category J (Science / academic prose), 80 files total |
| Passages used | 60 (50 main + 10 reserve) |
| Distinct source files | 49 |
| Characters (Han only) | 400–597, mean 481 |
| Distinct topic labels | 60 |
| Watermark-disabled / enabled outputs | 60 / 60 pairs |
| Total generation time | 44.5 min (21.5 disabled, 23.0 enabled) |