# -*- coding: utf-8 -*-
"""report_extras.py —— 汇总表之外的描述性数字"""
import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
it = list(csv.DictReader(open(BASE / "metrics_per_item.csv", encoding="utf-8-sig")))


def col(k):
    return np.array([float(r[k]) for r in it])


w = col("words_matched")
print(f"对齐词数：均值 {w.mean():.1f}｜中位 {np.median(w):.0f}｜区间 {w.min():.0f}-{w.max():.0f}")
print(f"原始词数 off：均值 {col('words_raw_off').mean():.1f}｜on：{col('words_raw_on').mean():.1f}")
trunc_on = sum(1 for r in it if float(r["words_raw_off"]) < float(r["words_raw_on"]))
trunc_off = sum(1 for r in it if float(r["words_raw_off"]) > float(r["words_raw_on"]))
print(f"被截断的条数：on 版 {trunc_on}｜off 版 {trunc_off}｜两者等长 {60 - trunc_on - trunc_off}")

for k, label in (("mattr", "MATTR"), ("tri", "Trigram"), ("g", "g-score")):
    o, n = col(f"{k}_off"), col(f"{k}_on")
    d = n - o
    print(f"{label}: off [{o.min():.4f},{o.max():.4f}] on [{n.min():.4f},{n.max():.4f}]｜"
          f"差值 升高 {int((d > 0).sum())} 条 / 下降 {int((d < 0).sum())} 条 / 持平 {int((d == 0).sum())}｜"
          f"|d| 中位 {np.median(np.abs(d)):.5f}")
