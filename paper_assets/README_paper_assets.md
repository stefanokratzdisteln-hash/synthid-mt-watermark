# 论文补充资产清单（对照 paper.pdf 的体检结果）

对照 `Downloads\paper.pdf`（7 页，含 Table 1、Table 2、Appendix A/B）逐项检查，以下是缺口与已备好的素材。

---

## 一、paper.pdf 现状

| 项目 | 现状 | 判断 |
|---|---|---|
| 结果表 | Table 1（检测）、Table 2（词汇） | 有，但**只有均值与差值 CI**，缺 SD、t、p、效应量 |
| 图 | 全文 0 张图 | **缺**，且正文已有 "item-level distributions were completely separated" 这类需要图示支撑的断言 |
| 参数可复现性 | Appendix A 只说"recorded"，无实际参数表 | **缺**，审稿人会直接问 keys、ngram_len、seed |
| 语料构成 | 只说 60 段 400–600 字 | **缺**量化的语料表 |
| 代码/数据可用性 | 无该声明 | **缺**，期刊与课程论文通常都要求 |
| 稳健性 | 只有配对 t 检验 | 可补一句正态性/非参数复核，成本极低 |

---

## 二、已生成的图（`paper_assets\`）

**fig1_paired_slope.png** — 配对斜率图，两联：(a) 正确密钥 g-score，(b) score gap。每条灰线是一个源文，红点为条件均值。
作用：把"60/60 全部升高 + 两条件不重叠"从文字断言变成可见证据。建议放 **§5.1 末尾，Table 1 之前或之后**。

**fig2_forest.png** — 四项结局的效应量森林图（黑=显著，灰=零）。零线为红色虚线。
作用：一张图同时交代"检测升、词汇不动"，正好是全文论证的主轴。建议放 **§5.2 开头**，或在 §6 Discussion 作为总结图。

规格：300 dpi PNG，单栏宽 6.6 / 6.8 英寸，Times 风格衬线字体，灰度友好（黑白打印仍可区分显著与不显著）。

---

## 三、已生成的表

**tables.md / tables.tex**
- `Table 1 (revised)`：四项结局 ×（Disabled M(SD)、Enabled M(SD)、ΔM[95%CI]、t(59)、p、d_z、升降条数、n）。补上了原表缺的 SD、t、p、d_z，并新增"↑/↓ 条数"一列——这一列正是你正文里"30 增 30 减""35 增 20 减 5 平"的来源，放进表里可省一段文字。
- `Table 2 (new)`：描述统计与区间（M、SD、range，两条件并排）。用途：直接支撑"两条件区间不重叠"。

**appendix_tables.md**
- `Table A1`：生成与检测的全部参数（模型、框架版本、解码参数、seed、keys 20 个整数、ngram_len=5、表大小 2^16、seed=0、context_history=1024、错误密钥生成方式、长度对齐规则、MATTR 实现）。
- `Table A2`：语料构成（LCMC J 类 80 篇中取 60 段 / 49 个来源篇目 / 汉字 400–597，均值 481 / 60 个学科标签 / 生成总耗时 44.5 min）。

---

## 四、可直接粘进论文的英文段落

### 4.1 Code and data availability（放 Appendix 或文末）

> **Code and data availability.** All generation, detection, and analysis scripts, together with the
> paired translation outputs, are archived with the project. `run_experiment_synthid.py` reproduces the
> paired translations from the cleaned source passages under a fixed seed; `compute_metrics.py`
> computes MATTR, the repeated trigram ratio, and the correct-key mean g-score; `compute_score_gap.py`
> computes the score gap against five incorrect keys; `validate_metrics.py` reports the negative-control
> checks; `make_paper_assets.py` regenerates Table 1 and Figures 1–2 directly from the per-item files.
> The per-item values underlying every reported statistic are in `metrics_per_item.csv` and
> `score_gap_results.csv`. Analyses used transformers 5.22.2 and torch 2.11.0 (CUDA 12.8) on a single
> consumer GPU; total generation time was 44.5 minutes.

### 4.2 Robustness footnote（放 Table 1 下方或 Appendix B）

> **Robustness.** Paired differences for the two detection outcomes were close to normal
> (Shapiro–Wilk *p* = .070 and .071) and their significance is unchanged under a Wilcoxon signed-rank
> test (*p* = 1.6 × 10⁻¹¹ for both). For the lexical outcomes the signed-rank test likewise returns
> non-significant results (MATTR *p* = .918; repeated trigram ratio *p* = .244), so the conclusions do
> not depend on the normality assumption. The repeated trigram ratio's differences deviate slightly from
> normality (*p* = .044); with *n* = 60 and a non-significant result this deviation does not affect
> interpretation.

### 4.3 图引用句

- Figure 1：`As Figure 1 shows, the change is uniform rather than aggregate: every pair moves upward and
  the two conditions do not overlap, so a single translation is classified correctly from its g-score
  alone (disabled max 0.5144 < enabled min 0.5246).`
- Figure 2：`Figure 2 places all four outcomes on a common scale: the two detection intervals lie far
  from zero while both lexical intervals straddle it, which is the divergence the study set out to test.`

---

## 五、还建议补但尚未生成的两项（需要的话我再做）

1. **例句对照附录（Appendix C）**：挑 1 段源文，并排给出 off / on 两版译文各 3–4 句，让读者直观看到"水印版读起来与无水印版无差别"。成本低、说服力强，尤其适合课程论文。
2. **逐条散点附录图**：`g-score(off)` vs `g-score(on)` 的对角线散点，所有点落在 y=x 之上。与 Figure 1 信息重复，只在版面宽裕时加。

## 六、一个需要你决定的口径问题

正文与 Table 1 现在写的是 **n = 60**（50 正式 + 10 备选）。Abstract 与 Methods 也说 60 段，口径自洽，无需改。若导师要求"只用 50 段正式样本"，替换数值为：g-score ΔM = +0.0397 [0.0365, 0.0429]，t(49) = 24.64，p = 2.9e−29，d_z = 3.48；score gap ΔM = +0.0401 [0.0366, 0.0436]，t(49) = 22.90，p = 7.9e−28，d_z = 3.24；词汇两项仍不显著。
