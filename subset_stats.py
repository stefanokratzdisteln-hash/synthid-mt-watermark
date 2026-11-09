# -*- coding: utf-8 -*-
"""subset_stats.py —— 只用 50 条正式样本（剔除 10 条备选）时的同一组统计量"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
groups = {r["id"]: r["group"] for r in csv.DictReader(open(BASE / "translation_results.csv", encoding="utf-8-sig"))}
formal = [i for i, g in groups.items() if g.strip() == "正式"]
print(f"正式样本 {len(formal)} 条，备选 {len(groups) - len(formal)} 条")

mi = {r["id"]: r for r in csv.DictReader(open(BASE / "metrics_per_item.csv", encoding="utf-8-sig"))}
sg = {r["id"]: r for r in csv.DictReader(open(BASE / "score_gap_results.csv", encoding="utf-8-sig"))}


def stats_block(label, off, on):
    d = np.array(on) - np.array(off)
    n = len(d)
    t, p = stats.ttest_rel(on, off)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"{label:26s} n={n} off={np.mean(off):.4f} on={np.mean(on):.4f} "
          f"Δ={d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] t={t:.2f} p={p:.2e} dz={d.mean()/d.std(ddof=1):.2f}")


stats_block("g-score (50 formal)", [float(mi[i]["g_off"]) for i in formal], [float(mi[i]["g_on"]) for i in formal])
stats_block("score gap (50 formal)", [float(sg[i]["gap_off"]) for i in formal], [float(sg[i]["gap_on"]) for i in formal])
