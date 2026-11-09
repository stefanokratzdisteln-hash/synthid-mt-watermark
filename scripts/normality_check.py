# -*- coding: utf-8 -*-
"""normality_check.py —— 为审稿预备：四项配对差的正态性检验 + 非参数复核"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

mi = list(csv.DictReader(open(DATA / "metrics_per_item.csv", encoding="utf-8-sig")))
sg = list(csv.DictReader(open(DATA / "score_gap_results.csv", encoding="utf-8-sig")))

cases = [("Correct-key mean g-score", mi, "g"), ("Score gap", sg, "gap"),
         ("MATTR (window 50)", mi, "mattr"), ("Repeated trigram ratio", mi, "tri")]

print(f"{'outcome':26s} {'Shapiro W':>10s} {'p':>8s} {'Wilcoxon p':>11s}  {'t-test p':>10s}")
for label, rows, k in cases:
    off = np.array([float(r[f"{k}_off"]) for r in rows])
    on = np.array([float(r[f"{k}_on"]) for r in rows])
    d = on - off
    W, p_sw = stats.shapiro(d)
    p_wx = stats.wilcoxon(on, off).pvalue
    p_t = stats.ttest_rel(on, off).pvalue
    print(f"{label:26s} {W:10.4f} {p_sw:8.4f} {p_wx:11.3g}  {p_t:10.3g}")
