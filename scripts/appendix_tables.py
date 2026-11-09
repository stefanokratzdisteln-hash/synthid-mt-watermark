# -*- coding: utf-8 -*-
"""appendix_tables.py —— 生成论文附录两张表：A1 参数表、A2 语料构成表

输入：data/translation_results.csv（只用到 id / group / topic / zh_source / sec_off / sec_on）
输出：results/appendix_tables.md
"""
import csv
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CSV_IN = DATA / "translation_results.csv"
OUT = RESULTS / "appendix_tables.md"

HAN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
N_SOURCE_FILES = 49          # 60 段来自 49 个 LCMC J 类篇目（篇目对应关系见论文 Appendix A）

rows = list(csv.DictReader(open(CSV_IN, encoding="utf-8-sig")))
chars = [len(HAN_RE.findall(r["zh_source"])) for r in rows]
groups = Counter(r["group"] for r in rows)
topics = [r["topic"] for r in rows]
minutes = (sum(float(r["sec_off"]) for r in rows) + sum(float(r["sec_on"]) for r in rows)) / 60
main = groups.get("正式", groups.get("main", 0))
reserve = groups.get("备选", groups.get("reserve", 0))
others = len(rows) - main - reserve

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
    "| Source corpus | LCMC category J (Science / academic prose), 80 files in total |",
    f"| Passages used | {len(rows)}"
    + (f" ({main} main + {reserve} reserve)" if reserve else "")
    + (f" ({others} unassigned)" if others else "")
    + " |",
    f"| Distinct source files | {N_SOURCE_FILES} |",
    f"| Characters (Han only) | {min(chars)}–{max(chars)}, mean {sum(chars) / len(chars):.0f} |",
    f"| Distinct topic labels | {len(set(topics))} |",
    f"| Paired translations | {len(rows)} watermark-disabled / {len(rows)} watermark-enabled |",
    f"| Total generation time | {minutes:.1f} min |",
]
RESULTS.mkdir(exist_ok=True)
OUT.write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
