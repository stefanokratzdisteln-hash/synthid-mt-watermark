# -*- coding: utf-8 -*-
"""compare_defs.py —— 三元组口径敏感性（含用户草稿中的算法）"""
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

rows = list(csv.DictReader(open(DATA / "translation_results.csv", encoding="utf-8-sig")))


def align(a, b):
    wa, wb = a.split(), b.split()
    n = min(len(wa), len(wb))
    return " ".join(wa[:n]), " ".join(wb[:n])


def tri(off_def, text):
    return off_def(text)


def def_a(text):   # 本表采用：重复类型所含 token / 全部 token
    w = re.findall(r"[a-z0-9']+", text.lower())
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    c = Counter(t)
    return sum(v for v in c.values() if v > 1) / len(t)


def def_b(text):   # 1 - 唯一 / 全部
    w = re.findall(r"[a-z0-9']+", text.lower())
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(t)) / len(t)


def def_c(text):   # 用户草稿：重复类型「个数」/ 全部三元组 token
    w = re.sub(r"[^a-z\s]", "", text.lower()).split()
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    c = Counter(t)
    return sum(1 for v in c.values() if v > 1) / len(t)


def def_d(text):   # 严格只留字母（用户草稿清洗），但按 token 计重
    w = re.sub(r"[^a-z\s]", "", text.lower()).split()
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    c = Counter(t)
    return sum(v for v in c.values() if v > 1) / len(t)


for fn, label in ((def_a, "A 重复类型token/全部(含数字撇号)"),
                  (def_b, "B 1-唯一/全部"),
                  (def_c, "C 重复类型个数/全部token（草稿口径）"),
                  (def_d, "D 重复类型token/全部(仅字母)")):
    off = [fn(align(r["en_watermark_off"], r["en_watermark_on"])[0]) for r in rows]
    on = [fn(align(r["en_watermark_off"], r["en_watermark_on"])[1]) for r in rows]
    d = np.array(on) - np.array(off)
    t, p = stats.ttest_rel(on, off)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"{label:34s} off={np.mean(off):.4f} on={np.mean(on):.4f} Δ={d.mean():+.5f} "
          f"CI=[{lo:+.5f},{hi:+.5f}] t={t:+.3f} p={p:.4f}")
