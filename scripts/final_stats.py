# -*- coding: utf-8 -*-
"""final_stats.py —— 输出写 Results 所需的完整统计量（含各组 SD 与精确 p）"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

it = list(csv.DictReader(open(DATA / "metrics_per_item.csv", encoding="utf-8-sig")))
sg = list(csv.DictReader(open(DATA / "score_gap_results.csv", encoding="utf-8-sig")))


def col(rows, k):
    return np.array([float(r[k]) for r in rows])


for k, label in (("g", "correct-key mean g-score"), ("gap", "score gap")):
    if k == "g":
        off, on, rows = col(it, "g_off"), col(it, "g_on"), it
    else:
        off, on, rows = col(sg, "gap_off"), col(sg, "gap_on"), sg
    d = on - off
    n = len(d)
    t, p = stats.ttest_rel(on, off)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"\n[{label}] n={n}")
    print(f"  off = {off.mean():.4f} (SD {off.std(ddof=1):.4f}), range {off.min():.4f}-{off.max():.4f}")
    print(f"  on  = {on.mean():.4f} (SD {on.std(ddof=1):.4f}), range {on.min():.4f}-{on.max():.4f}")
    print(f"  diff = {d.mean():+.4f} (SD {d.std(ddof=1):.5f}), 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  t({n-1}) = {t:.2f}, p = {p:.3e}, dz = {d.mean()/d.std(ddof=1):.2f}")
    print(f"  increased in {int((d>0).sum())}/{n} pairs；分离：off max {off.max():.4f} / on min {on.min():.4f}")
