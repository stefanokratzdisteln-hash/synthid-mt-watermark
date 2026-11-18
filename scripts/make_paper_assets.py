# -*- coding: utf-8 -*-
"""
Generate manuscript Figure 1 and the original paired-analysis supplements in results/.

The forest plot and companion tables report paired t intervals. Manuscript
Table 1 reports percentile bootstrap intervals from resampling source files.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

OUT = RESULTS
OUT.mkdir(exist_ok=True)

mi = list(csv.DictReader(open(DATA / "metrics_per_item.csv", encoding="utf-8-sig")))
sg = list(csv.DictReader(open(DATA / "score_gap_results.csv", encoding="utf-8-sig")))
group = {r["id"]: r["group"] for r in csv.DictReader(open(DATA / "translation_results.csv", encoding="utf-8-sig"))}

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
# Paired mean differences and t intervals in the original units of each outcome.
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
md = ["## Original paired t analysis of the four outcomes", "",
      "The 95% intervals below use the 60 passage-level differences (59 degrees of freedom). "
      "Table 1 in the manuscript uses a cluster bootstrap over the 49 source files.", "",
      "| Outcome | Disabled M (SD) | Enabled M (SD) | Paired ΔM [95% CI] | t(59) | p | d_z | Pairs ↑/↓ | n |",
      "|---|---|---|---|---|---|---|---|---|"]
for name, k, rows, fmt in OUTCOMES:
    s = S[k]
    md.append(
        f"| {name} | {s['m_off']:.4f} ({s['sd_off']:.4f}) | {s['m_on']:.4f} ({s['sd_on']:.4f}) | "
        f"{s['d']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] | {s['t']:.2f} | "
        f"{'< .001' if s['p'] < .001 else f'{s[chr(112)]:.3f}'} | {s['dz']:.2f} | "
        f"{s['up']}/{s['down']} | {s['n']} |")

md += ["", "> Note. d_z is the mean paired difference divided by its standard deviation. "
       "Pairs ↑/↓ gives the counts of increases and decreases, excluding ties.",
       "", "## Descriptive statistics and ranges per condition", "",
       "| Outcome | Disabled M | Disabled SD | Disabled range | Enabled M | Enabled SD | Enabled range |",
       "|---|---|---|---|---|---|---|"]
for name, k, rows, fmt in OUTCOMES:
    s = S[k]
    md.append(f"| {name} | {s['m_off']:.4f} | {s['sd_off']:.4f} | {s['min_off']:.4f} – {s['max_off']:.4f} | "
              f"{s['m_on']:.4f} | {s['sd_on']:.4f} | {s['min_on']:.4f} – {s['max_on']:.4f} |")
md += ["", "> Note. For both detection outcomes, the highest score with watermarking disabled "
       "fell below the lowest score with watermarking enabled in this sample."]
(OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")

# ---------------------------------------------------------------- LaTeX ----
tex = [r"""\begin{table}[t]
\centering
\caption{Original paired $t$ analysis of detection and lexical outcomes (60 pairs).}
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
Note. $\Delta M$ is the mean paired difference (watermark enabled $-$ disabled).
Intervals use the $t$ distribution with 59 degrees of freedom. Table 1 in the manuscript uses
a cluster bootstrap over the 49 source files. $d_z$ is the paired effect size.
\end{flushleft}
\end{table}""")
(OUT / "tables.tex").write_text("\n".join(tex), encoding="utf-8")

# ---------------------------------------------------------------- 图注 ----
caps = """# Figure captions

## Manuscript Figure 1. Paired Detection Scores With and Without Watermarking

Each gray line connects the two scores for one passage (60 pairs). Red dots show condition means.
Panel (a) shows the correct-key mean g-score. Panel (b) shows the gap between that score and the mean
under five incorrect key sets. The watermarked translation scored higher on both measures in all
60 pairs. [View Figure 1](fig1_paired_slope.png)

## Supplementary forest plot from the original paired t analysis

Dots show mean paired differences (watermark enabled minus disabled) in each outcome's original
units. The bars show 95% paired t intervals across 60 passage pairs. Both detection intervals lie
above the dashed zero line, and both lexical intervals include zero. Manuscript Table 1 reports
intervals from resampling whole source files. [View the forest plot](fig2_forest.png)
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
