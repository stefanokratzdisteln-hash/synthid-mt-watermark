# -*- coding: utf-8 -*-
"""
compute_metrics.py —— SynthID-Text 配对机器翻译实验的论文指标计算
=================================================================
输入：translation_results.csv（列含 en_watermark_off / en_watermark_on）
输出：metrics_per_item.csv（逐条指标）、metrics_summary.md（汇总表）

指标定义（全部在「长度对齐后」的文本上计算）：
  1. 长度对齐：按空白分词统计词数，把较长一条截断到与较短一条相同的词数
  2. MATTR(window=50)：转小写 → 移动窗口 50 词 → 各窗口 TTR 的平均（lexicalrichness 实现）
  3. Repeated Trigram Ratio：去标点、转小写 → 取词三元组 → 频次 >1 的三元组类型所含
     token 数 / 全部三元组 token 数（即 1 - 唯一三元组数/总三元组数 的「按词计」版本的下界）
  4. g-score：用与生成时完全相同的 SynthIDTextWatermarkLogitsProcessor（同一组 keys、
     同一 ngram_len、同一 sampling_table_seed）对文本重算 g-values，按
     「上下文重复掩码 × EOS 掩码」剔除无效位置后取均值。等价于 HF v4 的 mean_score
  5. 统计：配对差（On - Off）的均值、t 分布 95% CI、配对 t 检验
"""

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor

BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CSV_IN = DATA / "translation_results.csv"
# 分词器目录：默认按本地原始布局解析；公开仓库使用者可用环境变量 QWEN_MODEL_DIR 覆盖
MODEL_DIR = Path(os.environ.get(
    "QWEN_MODEL_DIR", str(ROOT.parent / "experiment" / "Qwen2.5-3B-Instruct")))
OUT_ITEM = DATA / "metrics_per_item.csv"
OUT_SUM = RESULTS / "metrics_summary.md"

# 与实验脚本 run_experiment_synthid.py 中完全一致的水印参数
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
        590, 639, 13, 715, 468, 990, 966, 226, 324, 585]
NGRAM_LEN = 5
TABLE_SIZE = 2 ** 16
TABLE_SEED = 0
MATTR_WINDOW = 50


# ---------------------------------------------------------------- 工具函数 ----
def align_by_words(a: str, b: str):
    """把较长一条截到与较短一条相同的词数，返回 (截断后 a, 截断后 b, 对齐词数)。"""
    wa, wb = a.split(), b.split()
    n = min(len(wa), len(wb))
    return " ".join(wa[:n]), " ".join(wb[:n]), n


def mattr(text: str) -> float:
    """MATTR(window=50)，先转小写。"""
    from lexicalrichness import LexicalRichness
    if len(text.split()) < MATTR_WINDOW:
        return float("nan")
    return LexicalRichness(text.lower()).mattr(window_size=MATTR_WINDOW)


def repeated_trigram_ratio(text: str) -> float:
    """重复三元组占比：频次>1 的三元组类型所含 token 数 / 全部三元组 token 数。"""
    import re
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < 3:
        return float("nan")
    tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    counts = Counter(tri)
    total = len(tri)
    in_repeated = sum(c for t, c in counts.items() if c > 1)
    return in_repeated / total


def mean_g_score(ids: torch.Tensor, processor: SynthIDTextWatermarkLogitsProcessor) -> float:
    """
    正确密钥下的平均 g-score。
    与 HF v4 的 mean_score 同构：g_values → (上下文重复掩码 × EOS 掩码) → 均值。
    """
    ids = ids.unsqueeze(0)                                  # (1, L)
    g = processor.compute_g_values(ids)                      # (1, L-ngram+1, depth)
    eos_mask = processor.compute_eos_token_mask(ids, eos_token_id=TOKENIZER.eos_token_id)
    eos_mask = eos_mask[:, processor.ngram_len - 1:]         # 与 g 的序列长度对齐
    rep_mask = processor.compute_context_repetition_mask(ids)
    mask = (eos_mask * rep_mask).float()                     # (1, L-ngram+1)
    if mask.sum() == 0:
        return float("nan")
    # 对每个 (位置, depth) 求 g 的均值，权重为 mask（mask 只与位置有关，故等价于按位置加权后整体平均）
    weighted = (g.float() * mask.unsqueeze(-1)).sum()
    denom = mask.sum() * g.shape[-1]
    return float(weighted / denom)


def fmt(x, nd=4):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- 主流程 ----
def main():
    global TOKENIZER
    rows = list(csv.DictReader(open(CSV_IN, encoding="utf-8-sig")))
    print(f"读入 {len(rows)} 条")

    TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=NGRAM_LEN,
        keys=KEYS,
        sampling_table_size=TABLE_SIZE,
        sampling_table_seed=TABLE_SEED,
        context_history_size=1024,
        device=device,
    )
    print(f"水印 processor 设备：{device}｜depth={len(KEYS)}｜ngram_len={NGRAM_LEN}")

    items = []
    for r in rows:
        off_raw, on_raw = r["en_watermark_off"], r["en_watermark_on"]
        off, on, n = align_by_words(off_raw, on_raw)

        rec = {
            "id": r["id"],
            "words_raw_off": len(off_raw.split()),
            "words_raw_on": len(on_raw.split()),
            "words_matched": n,
            "mattr_off": mattr(off),
            "mattr_on": mattr(on),
            "tri_off": repeated_trigram_ratio(off),
            "tri_on": repeated_trigram_ratio(on),
        }
        for tag, text in (("off", off), ("on", on)):
            ids = TOKENIZER(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            rec[f"g_{tag}"] = mean_g_score(ids, processor)
        items.append(rec)

    # 只保留四项指标全部算出来的样本（有效配对）
    metrics = [
        ("MATTR (window=50)", "mattr_off", "mattr_on"),
        ("Repeated Trigram Ratio", "tri_off", "tri_on"),
        ("g-score（正确密钥）", "g_off", "g_on"),
    ]

    summary = {}
    for name, k_off, k_on in metrics:
        off_vals = np.array([it[k_off] for it in items], dtype=float)
        on_vals = np.array([it[k_on] for it in items], dtype=float)
        ok = ~(np.isnan(off_vals) | np.isnan(on_vals))
        n = int(ok.sum())
        d = on_vals[ok] - off_vals[ok]
        m_off, m_on, m_d = float(np.mean(off_vals[ok])), float(np.mean(on_vals[ok])), float(np.mean(d))
        sd = float(np.std(d, ddof=1))
        se = sd / math.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci = (m_d - t_crit * se, m_d + t_crit * se)
        t_val, p_val = stats.ttest_rel(on_vals[ok], off_vals[ok])
        summary[k_off] = dict(name=name, n=n, m_off=m_off, m_on=m_on,
                              d=m_d, ci=ci, t=float(t_val), p=float(p_val),
                              sd=sd, dz=m_d / sd if sd else float("nan"))

    # 逐条明细
    with open(OUT_ITEM, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        w.writeheader()
        w.writerows(items)

    # 汇总 markdown
    lines = [
        "# SynthID-Text 配对机器翻译实验 · 指标汇总（长度对齐后）",
        "",
        f"- 样本：{len(items)} 条配对，长度对齐后平均 {np.mean([it['words_matched'] for it in items]):.0f} 词",
        f"- 对齐方式：按空白分词，较长一条截断至与较短一条相同的词数",
        f"- g-score：与生成时相同的 keys（depth={len(KEYS)}）、ngram_len={NGRAM_LEN}、"
        f"sampling_table_seed={TABLE_SEED}，剔除重复上下文与 EOS 位置后取均值",
        "",
        "| 指标名称 | 无水印均值 | 有水印均值 | 配对差值均值 | 95% CI | t 值 | p 值 | 有效配对数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, s in summary.items():
        lines.append(
            f"| {s['name']} | {fmt(s['m_off'])} | {fmt(s['m_on'])} | {fmt(s['d'], 5)} | "
            f"[{fmt(s['ci'][0], 5)}, {fmt(s['ci'][1], 5)}] | {fmt(s['t'], 3)} | {fmt(s['p'], 4)} | {s['n']} |"
        )
    lines += ["", "注：配对差值 = 有水印 − 无水印；CI 由 t 分布（df = n−1）构造；检验为双侧配对样本 t 检验。",
              f"效应量（Cohen's dz）：" + "；".join(f"{s['name']} = {fmt(s['dz'], 3)}" for s in summary.values()) + "。"]
    OUT_SUM.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines[5:]) )
    print(f"\n明细：{OUT_ITEM}\n汇总：{OUT_SUM}")


if __name__ == "__main__":
    main()
