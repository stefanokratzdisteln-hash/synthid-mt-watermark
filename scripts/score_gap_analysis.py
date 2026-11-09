# -*- coding: utf-8 -*-
"""
score_gap_analysis.py —— 次要指标 Score Gap 的配对分析（可运行版）

定义与 Downloads\\compute_score_gap.py 一致：
    Score Gap = 正确密钥下的平均 g-score − 5 组错误密钥下的平均 g-score
    错误密钥 = torch.manual_seed(seed) 后 randint(0, 10000, (20,))，seed = 101/202/303/404/505

与主指标口径对齐：
  * 文本先 length-match（截断较长一条），使 off/on 等长，避免长度混杂
  * g-score 的掩码与主表完全一致（上下文重复掩码 × EOS 掩码）
"""

import csv
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL_DIR = ROOT.parent / "experiment" / "Qwen2.5-3B-Instruct"
CSV_PATH = DATA / "translation_results.csv"
OUT_PATH = DATA / "score_gap_results.csv"

CORRECT_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
                590, 639, 13, 715, 468, 990, 966, 226, 324, 585]


def wrong_keys(seed):
    torch.manual_seed(seed)
    return torch.randint(0, 10000, (20,)).tolist()


WRONG_KEYS = [wrong_keys(s) for s in (101, 202, 303, 404, 505)]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)


def processor(keys):
    # transformers 5.x 的 device 为必填参数
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=5, keys=keys, sampling_table_size=2 ** 16,
        sampling_table_seed=0, context_history_size=1024, device=dev)


PROC_CORRECT = processor(CORRECT_KEYS)
PROC_WRONG = [processor(k) for k in WRONG_KEYS]


def g_score(text, proc):
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev).unsqueeze(0)
    g = proc.compute_g_values(ids)
    m = proc.compute_eos_token_mask(ids, eos_token_id=tok.eos_token_id)[:, proc.ngram_len - 1:]
    m = (m * proc.compute_context_repetition_mask(ids)).float()
    if m.sum() == 0:
        return float("nan")
    return float((g.float() * m.unsqueeze(-1)).sum() / (m.sum() * g.shape[-1]))


def align(a, b):
    wa, wb = a.split(), b.split()
    n = min(len(wa), len(wb))
    return " ".join(wa[:n]), " ".join(wb[:n])


rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
out = []
for i, r in enumerate(rows, 1):
    off, on = align(r["en_watermark_off"], r["en_watermark_on"])
    c_off, c_on = g_score(off, PROC_CORRECT), g_score(on, PROC_CORRECT)
    w_off = [g_score(off, p) for p in PROC_WRONG]
    w_on = [g_score(on, p) for p in PROC_WRONG]
    rec = {
        "id": r["id"],
        "g_off_correct": c_off,
        "g_on_correct": c_on,
        "g_off_wrong_mean": float(np.mean(w_off)),
        "g_on_wrong_mean": float(np.mean(w_on)),
        "gap_off": c_off - float(np.mean(w_off)),
        "gap_on": c_on - float(np.mean(w_on)),
    }
    rec["gap_diff"] = rec["gap_on"] - rec["gap_off"]
    out.append(rec)
    if i % 10 == 0:
        print(f"  {i}/{len(rows)}")

with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

gap_off = np.array([o["gap_off"] for o in out])
gap_on = np.array([o["gap_on"] for o in out])
d = gap_on - gap_off
n = len(d)
m_off, m_on, m_d = float(gap_off.mean()), float(gap_on.mean()), float(d.mean())
sd, se = float(d.std(ddof=1)), float(d.std(ddof=1) / np.sqrt(n))
lo, hi = stats.t.interval(0.95, n - 1, loc=m_d, scale=se)
t_stat, p_val = stats.ttest_rel(gap_on, gap_off)

print("\n=== Score Gap 配对分析（length-matched）===")
print(f"无水印 gap 均值 = {m_off:.5f}（SD {gap_off.std(ddof=1):.5f}，区间 {gap_off.min():.5f}~{gap_off.max():.5f}）")
print(f"有水印 gap 均值 = {m_on:.5f}（SD {gap_on.std(ddof=1):.5f}，区间 {gap_on.min():.5f}~{gap_on.max():.5f}）")
print(f"配对差值 = {m_d:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]")
print(f"t = {t_stat:.4f}   p = {p_val:.4g}   n = {n}   dz = {m_d / sd:.3f}")
print(f"有水印 gap > 无水印 gap 的条数：{int((d > 0).sum())}/{n}")
print(f"gap_on 最小值 {gap_on.min():.5f} vs gap_off 最大值 {gap_off.max():.5f}（分布是否分离）")
print(f"明细：{OUT_PATH}")
