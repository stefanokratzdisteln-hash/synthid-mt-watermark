# -*- coding: utf-8 -*-
"""
validate_metrics.py —— 指标口径校验
1) g-score 阴性对照：用「值不同的另一组密钥」而非密钥排列（排列不改变按深度平均的 g）
2) 重复三元组两种口径：结论是否稳健
"""
import csv
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get(
    "QWEN_MODEL_DIR", str(BASE.parent / "experiment" / "Qwen2.5-3B-Instruct")))
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
        590, 639, 13, 715, 468, 990, 966, 226, 324, 585]
# 真正不同的密钥值：整体 +37 平移（不是 KEYS 的排列）
KEYS_ALT = [k + 37 for k in KEYS]
# 另加一组随机密钥
rng = np.random.default_rng(20261110)
KEYS_RND = rng.integers(1, 1000, size=len(KEYS)).tolist()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)


def make_proc(keys):
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=5, keys=keys, sampling_table_size=2 ** 16,
        sampling_table_seed=0, context_history_size=1024, device=dev)


PROCS = {"正确": make_proc(KEYS), "平移+37": make_proc(KEYS_ALT), "随机": make_proc(KEYS_RND)}


def g_score(text, proc):
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev).unsqueeze(0)
    g = proc.compute_g_values(ids)
    m = proc.compute_eos_token_mask(ids, eos_token_id=tok.eos_token_id)[:, proc.ngram_len - 1:]
    m = (m * proc.compute_context_repetition_mask(ids)).float()
    return float((g.float() * m.unsqueeze(-1)).sum() / (m.sum() * g.shape[-1]))


def align(a, b):
    wa, wb = a.split(), b.split()
    n = min(len(wa), len(wb))
    return " ".join(wa[:n]), " ".join(wb[:n])


def tri_a(text):   # 重复类型所含 token / 全部 token
    w = re.findall(r"[a-z0-9']+", text.lower())
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    c = Counter(t)
    return sum(v for v in c.values() if v > 1) / len(t)


def tri_b(text):   # 1 - 唯一 / 全部
    w = re.findall(r"[a-z0-9']+", text.lower())
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(t)) / len(t)


def report(name, off, on):
    d = np.array(on) - np.array(off)
    t, p = stats.ttest_rel(on, off)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"{name:28s} off={np.mean(off):.4f} on={np.mean(on):.4f} Δ={d.mean():+.5f} "
          f"CI=[{lo:+.5f},{hi:+.5f}] t={t:+.3f} p={p:.4g}")


rows = list(csv.DictReader(open(BASE / "translation_results.csv", encoding="utf-8-sig")))
res = {k: {"off": [], "on": []} for k in PROCS}
tri_a_off, tri_a_on, tri_b_off, tri_b_on = [], [], [], []
for r in rows:
    off, on = align(r["en_watermark_off"], r["en_watermark_on"])
    for k, proc in PROCS.items():
        res[k]["off"].append(g_score(off, proc))
        res[k]["on"].append(g_score(on, proc))
    tri_a_off.append(tri_a(off)); tri_a_on.append(tri_a(on))
    tri_b_off.append(tri_b(off)); tri_b_on.append(tri_b(on))

print("=== g-score：正确密钥 vs 错误密钥（值不同）===")
for k in PROCS:
    report(f"{k}密钥 (off vs on)", res[k]["off"], res[k]["on"])
for k in ("平移+37", "随机"):
    n_on_above = sum(1 for v in res[k]["on"] if v > 0.5)
    n_off_above = sum(1 for v in res[k]["off"] if v > 0.5)
    print(f"  {k}密钥：on 版 >0.5 的条数 {n_on_above}/60；off 版 {n_off_above}/60")
print(f"  正确密钥：on 版 >0.5 的条数 {sum(1 for v in res['正确']['on'] if v > 0.5)}/60；"
      f"off 版 {sum(1 for v in res['正确']['off'] if v > 0.5)}/60")

print("\n=== 重复三元组两种口径 ===")
report("口径A 重复类型token/全部", tri_a_off, tri_a_on)
report("口径B 1-唯一/全部", tri_b_off, tri_b_on)
