# -*- coding: utf-8 -*-
"""appendix_tables.py —— 生成论文附录所需的两张配置表（A1 参数、A2 语料构成）"""
import csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
CORPUS = Path(r"C:\Users\KRATZDISTELN\Doubao\chats\2026-11-09\new-chat\final_60_texts.csv")
OUT = BASE / "paper_assets" / "appendix_tables.md"

rows = list(csv.DictReader(open(CORPUS, encoding="utf-8-sig")))
files = [r["来源文件序号"] for r in rows]
chars = [int(r["纯汉字字数"]) for r in rows]
groups = Counter(r["组别"] for r in rows)
topics = [r["学科主题"] for r in rows]

md = [
    "## Table A1. Generation and detection configuration",
    "",
    "| Component | Value |",
    "|---|---|",
    "| Model | Qwen2.5-3B-Instruct (local weights, bf16, no quantization) |",
    "| Framework | transformers 5.22.2; torch 2.11.0+cu128 |",
    "| Decoding | `max_new_tokens=1024`, `temperature=1.0`, `do_sample=True`, `top_p=0.95` |",
    "| Random seed | 42 (identical in both conditions) |",
    "| Watermark | SynthID-Text via `SynthIDTextWatermarkingConfig` |",
    "| Keys (depth 20) | 654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585 |",
    "| `ngram_len` | 5 |",
    "| `sampling_table_size` | 2^16 = 65536 |",
    "| `sampling_table_seed` | 0 |",
    "| `context_history_size` | 1024 |",
    "| Detection statistic | mean g-score: `compute_g_values()` averaged over all valid (position, depth) cells, excluding repeated-context and EOS positions |",
    "| Incorrect keys | five key sets drawn as `randint(0, 10000, 20)` with seeds 101 / 202 / 303 / 404 / 505 |",
    "| Length matching | longer translation truncated to the shorter one's whitespace-token count |",
    "| MATTR | `lexicalrichness` 0.5.1, window = 50, lowercased |",
    "",
    "## Table A2. Corpus composition",
    "",
    "| Property | Value |",
    "|---|---|",
    f"| Source corpus | LCMC category J (Science / academic prose), 80 files total |",
    f"| Passages used | {len(rows)} ({groups.get('正式', 0)} main + {groups.get('备选', 0)} reserve) |",
    f"| Distinct source files | {len(set(files))} |",
    f"| Characters (Han only) | {min(chars)}–{max(chars)}, mean {sum(chars) / len(chars):.0f} |",
    f"| Distinct topic labels | {len(set(topics))} |",
    f"| Watermark-disabled / enabled outputs | {len(rows)} / {len(rows)} pairs |",
    f"| Total generation time | 44.5 min (21.5 disabled, 23.0 enabled) |",
]
OUT.write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
