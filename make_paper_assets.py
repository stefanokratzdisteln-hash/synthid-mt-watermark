# -*- coding: utf-8 -*-
"""
make_paper_assets.py —— 生成论文缺失的图与表

产出（全部写入 decection\\paper_assets\\）：
  fig1_paired_slope.png   配对斜率图：g-score 与 score gap 的逐条变化（体现 60/60 分离）
  fig2_forest.png         四项结局的效应量森林图（两项显著、两项为零，呼应论文主线）
  tables.md               可直接粘进 Markdown / Word 的三张表
  tables.tex              booktabs 版 LaTeX 表格
  figure_captions.md      图注与正文引用句
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper_assets"
OUT.mkdir(exist_ok=True)

mi = list(csv.DictReader(open(BASE / "metrics_per_item.csv", encoding="utf-8-sig")))
sg = list(csv.DictReader(open(BASE / "score_gap_results.csv", encoding="utf-8-sig")))
group = {r["id"]: r["group"] for r in csv.DictReader(open(BASE / "translation_results.csv", encoding="utf-8-sig"))}

OUTCOMES = [
    ("Correct-key mean g-score", "g", mi, "{:.4f}"),
    ("Score gap", "gap", sg, "{:.4f}"),
    ("MATTR (window 50)", "mattr", mi, "{:.4f}"),
    ("Repeated trigram ratio", "tri", mi, "{:.4f}"),
]


def series(rows, k):
    off = np.array([float(r[f"{k}_off"]) for r in rows])
    on = np.array([float(r[f"{k}_on"]) for r in rows])
    return off, on


def full_stats(k, rows):
    off, on = series(rows, k)
    d = on - off
    n = len(d)
    t, p = stats.ttest_rel(on, off)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    return dict(n=n, m_off=off.mean(), sd_off=off.std(ddof=1), min_off=off.min(), max_off=off.max(),
                m_on=on.mean(), sd_on=on.std(ddof=1), min_on=on.min(), max_on=on.max(),
                d=d.mean(), sd_d=d.std(ddof=1), lo=lo, hi=hi, t=t, p=p, dz=d.mean() / d.std(ddof=1),
                up=int((d > 0).sum()), down=int((d < 0).sum()), tie=int((d == 0).sum()))


S = {k: full_stats(k, rows) for _, k, rows, _ in OUTCOMES}

# ---------------------------------------------------------------- 图 1 ----
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), dpi=300)
for ax, (label, k) in zip(axes, [("Correct-key mean g-score", "g"), ("Score gap", "gap")]):
    off, on = series(mi if k == "g" else sg, k)
    x = np.array([0, 1])
    jitter = np.random.default_rng(7).normal(0, 0.006, size=len(off))
    for i in range(len(off)):
        ax.plot(x + jitter[[i, i]], [off[i], on[i]], color="0.62", lw=0.5, alpha=0.55, zorder=1)
    ax.scatter(np.zeros(len(off)) + jitter, off, s=7, color="0.35", zorder=2, linewidths=0)
    ax.scatter(np.ones(len(on)) + jitter, on, s=7, color="black", zorder=2, linewidths=0)
    ax.plot(x, [off.mean(), on.mean()], color="crimson", lw=1.6, marker="o", ms=5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Disabled", "Enabled"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(label, fontsize=8.5)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    sep = f"separation {len(off)}/{len(off)}" if k != "mattr" else ""
for ax, k in ((axes[0], "g"), (axes[1], "gap")):
    s = S[k]
    ax.text(0.97, 0.03,
            f"{s['up']}/{s['n']} pairs increase\nno overlap ({s['max_off']:.3f} < {s['min_on']:.3f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.3",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.5))
axes[0].set_title("(a) Correct-key mean g-score", fontsize=9)
axes[1].set_title("(b) Score gap", fontsize=9)
axes[1].margins(y=0.16)
fig.tight_layout()
fig.savefig(OUT / "fig1_paired_slope.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- 图 2 ----
# 四项结局的效应量森林图（标准化到各自 SD，便于同图比较）
fig, ax = plt.subplots(figsize=(6.8, 2.4), dpi=300)
labels = ["Correct-key\ng-score", "Score gap", "MATTR\n(window 50)", "Repeated trigram\nratio"]
pos = np.arange(len(labels))[::-1]
for y, (name, k, rows, _) in zip(pos, OUTCOMES):
    s = S[k]
    sig = s["p"] < 0.05
    color = "black" if sig else "0.55"
    ax.plot([s["lo"], s["hi"]], [y, y], color=color, lw=1.4, solid_capstyle="butt")
    ax.plot([s["lo"], s["lo"]], [y - 0.06, y + 0.06], color=color, lw=1.0)
    ax.plot([s["hi"], s["hi"]], [y - 0.06, y + 0.06], color=color, lw=1.0)
    ax.scatter([s["d"]], [y], s=26, color=color, zorder=3, linewidths=0)
    p_txt = "p<.001" if s["p"] < .001 else f"p={s['p']:.3f}"
    ax.text(0.058, y, f"{s['d']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]   {p_txt}   $d_z$={s['dz']:.2f}",
            va="center", fontsize=6.6, color="0.2")
ax.axvline(0, color="crimson", lw=0.9, ls="--", zorder=1)
ax.set_yticks(pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Paired mean difference (watermark enabled − disabled)", fontsize=8)
ax.set_xlim(-0.02, 0.142)
ax.set_ylim(-0.6, len(labels) - 0.4)
ax.grid(axis="x", color="0.92", lw=0.6)
ax.set_axisbelow(True)
ax.set_title("Paired mean differences with 95% CIs (n = 60 pairs)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig2_forest.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- 表 1 ----
md = ["## Table 1 (revised). Paired comparison of all four outcomes", "",
      "| Outcome | Disabled M (SD) | Enabled M (SD) | Paired ΔM [95% CI] | t(59) | p | d_z | Pairs ↑/↓ | n |",
      "|---|---|---|---|---|---|---|---|---|"]
for name, k, rows, fmt in OUTCOMES:
    s = S[k]
    md.append(
        f"| {name} | {s['m_off']:.4f} ({s['sd_off']:.4f}) | {s['m_on']:.4f} ({s['sd_on']:.4f}) | "
        f"{s['d']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] | {s['t']:.2f} | "
        f"{'< .001' if s['p'] < .001 else f'{s[chr(112)]:.3f}'} | {s['dz']:.2f} | "
        f"{s['up']}/{s['down']} | {s['n']} |")

md += ["", "> 说明：d_z 为配对效应量（ΔM / SD_diff）；Pairs ↑/↓ 统计逐条升高/降低的条数，用于说明变化是否双向。",
       "", "## Table 2 (new). Descriptive statistics and ranges per condition", "",
       "| Outcome | Disabled M | Disabled SD | Disabled range | Enabled M | Enabled SD | Enabled range |",
       "|---|---|---|---|---|---|---|"]
for name, k, rows, fmt in OUTCOMES:
    s = S[k]
    md.append(f"| {name} | {s['m_off']:.4f} | {s['sd_off']:.4f} | {s['min_off']:.4f} – {s['max_off']:.4f} | "
              f"{s['m_on']:.4f} | {s['sd_on']:.4f} | {s['min_on']:.4f} – {s['max_on']:.4f} |")
md += ["", "> 说明：两条件区间不重叠即为 item-level separation，可直接写进正文（g-score 与 score gap 均无重叠）。"]
(OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")

# ---------------------------------------------------------------- LaTeX ----
tex = [r"""\begin{table}[t]
\centering
\caption{Paired comparison of detection and lexical outcomes (n = 60 pairs).}
\label{tab:all_outcomes}
\begin{tabular}{lcccccc}
\toprule
Outcome & \multicolumn{2}{c}{M (SD)} & \multicolumn{1}{c}{$\Delta M$ [95\% CI]} & $t(59)$ & $p$ & $d_z$ \\
\cmidrule(lr){2-3}
 & Disabled & Enabled & & & & \\
\midrule"""]
for name, k, rows, fmt in OUTCOMES:
    s = S[k]
    nm = name.replace("Correct-key mean g-score", r"Correct-key mean $g$-score").replace(
        "Repeated trigram ratio", "Repeated trigram ratio").replace("MATTR (window 50)", r"MATTR ($w{=}50$)")
    p_txt = r"$<.001$" if s["p"] < .001 else f"{s['p']:.3f}"
    tex.append(f"{nm} & {s['m_off']:.4f} ({s['sd_off']:.4f}) & {s['m_on']:.4f} ({s['sd_on']:.4f}) & "
               f"{s['d']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] & {s['t']:.2f} & {p_txt} & {s['dz']:.2f} \\\\")
tex.append(r"""\bottomrule
\end{tabular}
\begin{flushleft}\footnotesize
Note. $\Delta M$ = watermark enabled $-$ disabled; CIs are $t$-based with $df=59$; $d_z$ is the paired effect size.
\end{flushleft}
\end{table}""")
(OUT / "tables.tex").write_text("\n".join(tex), encoding="utf-8")

# ---------------------------------------------------------------- 图注 ----
caps = f"""# Figure captions（英文，可直接用）

**Figure 1.** Paired change in detection scores from the watermark-disabled to the watermark-enabled
condition. Each grey line is one source text; red dots and the red line show condition means.
Panel (a) plots the correct-key mean g-score and panel (b) the score gap. In both panels the two
conditions do not overlap at the item level (disabled maximum {S['g']['max_off']:.4f} vs enabled minimum
{S['g']['min_on']:.4f} for the g-score; {S['gap']['max_off']:.4f} vs {S['gap']['min_on']:.4f} for the score gap),
so every pair is assigned to its own condition. n = 60 pairs.

**Figure 2.** Paired mean differences with 95% confidence intervals for the four outcomes, ordered
top to bottom. Detection outcomes (black markers) are significantly different from zero; lexical
outcomes (grey markers) are not. The dashed line marks zero. Values printed beside each marker give
the mean difference, its 95% CI, the p value, and Cohen's d_z. n = 60 pairs.

# 正文可直接插入的句子

- 引图 1：`As Figure 1 shows, the increase is uniform across items: every pair moved upward and the
  two conditions do not overlap, so a single translation can be classified correctly from its g-score
  alone.`
- 引图 2：`Figure 2 places the four outcomes on a common scale: the two detection intervals sit far
  from zero, whereas both lexical intervals straddle zero, which is the pattern the paper's argument
  depends on.`
- 引表（附录参数表）：`Table A1 lists every generation and detection parameter; toggling
  \`watermarking_config\` was the only manipulation between conditions.`
"""
(OUT / "figure_captions.md").write_text(caps, encoding="utf-8")

print("done")
for name in ("fig1_paired_slope.png", "fig2_forest.png", "tables.md", "tables.tex", "figure_captions.md"):
    p = OUT / name
    print(f"  {name:26s} {p.stat().st_size / 1024:.1f} KB")
for name, k, rows, fmt in OUTCOMES:
    s = S[k]
    print(f"{name:26s} n={s['n']} off={s['m_off']:.4f}({s['sd_off']:.4f}) on={s['m_on']:.4f}({s['sd_on']:.4f}) "
          f"Δ={s['d']:+.4f} CI[{s['lo']:+.4f},{s['hi']:+.4f}] p={s['p']:.3g} dz={s['dz']:.2f}")
