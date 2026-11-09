# -*- coding: utf-8 -*-
"""
计算次要指标：Score Gap = 正确密钥 g-score - 5 个错误密钥平均 g-score

【修订记录 2026-11-10】为在 transformers 5.22.2 上可运行，改动仅 4 处，定义与统计流程未变：
  1. TOKENIZER_PATH 改为绝对路径并 local_files_only=True（原 "./Qwen2.5-3B-Instruct" 在 v5 被当作 repo id 直接报错）
  2. SynthIDTextWatermarkLogitsProcessor 补 device 参数（v5 起为必填），并改为循环外只建一次
  3. compute_eos_token_mask 补 eos_token_id，并按 [:, ngram_len-1:] 与 g_values 的序列长度对齐
  4. 文本先 length-match（截断较长一条）再算 g-score，与主指标 MATTR/Trigram 的口径一致
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

# ============================================================
# 1. 配置区
# ============================================================
BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CSV_PATH = DATA / "translation_results.csv"          # 你的原始结果文件
METRICS_PATH = DATA / "metrics_per_item.csv"         # 之前算出的正确密钥 g-score
OUTPUT_PATH = DATA / "score_gap_results.csv"
TOKENIZER_PATH = Path(os.environ.get(
    "QWEN_MODEL_DIR", str(ROOT.parent / "experiment" / "Qwen2.5-3B-Instruct")))   # 绝对路径，v5 需 local_files_only

# 你的正确密钥 (必须和生成时一致)
CORRECT_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
                590, 639, 13, 715, 468, 990, 966, 226, 324, 585]


# 生成 5 组随机错误密钥 (保持同样的长度 20)
def generate_random_keys(seed):
    torch.manual_seed(seed)
    return torch.randint(0, 10000, (20,)).tolist()


WRONG_KEYS_LIST = [generate_random_keys(s) for s in (101, 202, 303, 404, 505)]

# ============================================================
# 2. 核心计算函数
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)


def make_processor(keys):
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=5,
        keys=keys,
        sampling_table_size=2 ** 16,
        sampling_table_seed=0,
        context_history_size=1024,
        device=DEVICE,
    )


PROC_CORRECT = make_processor(CORRECT_KEYS)
PROC_WRONG = [make_processor(k) for k in WRONG_KEYS_LIST]


def compute_g_score(text, processor):
    """使用给定的 processor 计算文本的平均 g-score（正确密钥 / 错误密钥通用）"""
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)

    g_values = processor.compute_g_values(input_ids)                     # (1, L-4, 20)
    mask = processor.compute_context_repetition_mask(input_ids)
    eos = processor.compute_eos_token_mask(input_ids, eos_token_id=tokenizer.eos_token_id)
    mask = (mask * eos[:, processor.ngram_len - 1:]).float()             # 与 g 序列长度对齐

    valid_count = mask.sum() * g_values.shape[-1]
    if valid_count == 0:
        return 0.5                                                       # 极端情况返回期望值
    return float((g_values.float() * mask.unsqueeze(-1)).sum() / valid_count)


def align(text_a, text_b):
    """length-match：把较长一条截断到较短一条的词数"""
    wa, wb = text_a.split(), text_b.split()
    n = min(len(wa), len(wb))
    return " ".join(wa[:n]), " ".join(wb[:n])


def main():
    print("正在加载数据...")
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    metrics = {r["id"]: r for r in csv.DictReader(open(METRICS_PATH, encoding="utf-8-sig"))}

    results = []
    print(f"开始计算 Score Gap (共 {len(rows)} 条，每条 5 组错误密钥)...")
    for i, row in enumerate(rows, 1):
        text_off, text_on = align(row["en_watermark_off"], row["en_watermark_on"])

        # 正确密钥的得分：直接重算，保证与错误密钥算在同一条（已对齐）文本上
        correct_off = compute_g_score(text_off, PROC_CORRECT)
        correct_on = compute_g_score(text_on, PROC_CORRECT)

        wrong_scores_off = [compute_g_score(text_off, p) for p in PROC_WRONG]
        wrong_scores_on = [compute_g_score(text_on, p) for p in PROC_WRONG]

        avg_wrong_off = sum(wrong_scores_off) / len(wrong_scores_off)
        avg_wrong_on = sum(wrong_scores_on) / len(wrong_scores_on)

        results.append({
            "id": row["id"],
            "g_off_correct": correct_off,
            "g_on_correct": correct_on,
            "g_off_wrong_mean": avg_wrong_off,
            "g_on_wrong_mean": avg_wrong_on,
            "gap_off": correct_off - avg_wrong_off,
            "gap_on": correct_on - avg_wrong_on,
            "gap_diff": (correct_on - avg_wrong_on) - (correct_off - avg_wrong_off),
            "g_off_cached": float(metrics[row["id"]]["g_off"]),   # 与主表交叉核对
            "g_on_cached": float(metrics[row["id"]]["g_on"]),
        })
        if i % 10 == 0:
            print(f"已处理 {i}/{len(rows)}")

    fieldnames = ["id", "g_off_correct", "g_on_correct", "g_off_wrong_mean", "g_on_wrong_mean",
                  "gap_off", "gap_on", "gap_diff"]
    with open(OUTPUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows([{k: r[k] for k in fieldnames} for r in results])

    # 与主表 g-score 的一致性核对
    d_off = max(abs(r["g_off_correct"] - r["g_off_cached"]) for r in results)
    d_on = max(abs(r["g_on_correct"] - r["g_on_cached"]) for r in results)
    print(f"\n与主表 g-score 的最大偏差：off {d_off:.2e}｜on {d_on:.2e}（应≈0，说明两处算法一致）")

    # ============================================================
    # 3. 统计检验与打印
    # ============================================================
    gap_off = np.array([r["gap_off"] for r in results])
    gap_on = np.array([r["gap_on"] for r in results])
    diffs = np.array([r["gap_diff"] for r in results])
    n = len(diffs)

    print("\n" + "=" * 50)
    print("Score Gap 统计结果 (用于补充到论文中)")
    print("=" * 50)
    ci = stats.t.interval(0.95, df=n - 1, loc=diffs.mean(), scale=stats.sem(diffs))
    t_stat, p_val = stats.ttest_rel(gap_on, gap_off)

    print(f"无水印 Score Gap 均值: {gap_off.mean():.5f}（SD {gap_off.std(ddof=1):.5f}）")
    print(f"有水印 Score Gap 均值: {gap_on.mean():.5f}（SD {gap_on.std(ddof=1):.5f}）")
    print(f"配对差值 (On - Off) 均值 [95% CI]: {diffs.mean():.5f} [{ci[0]:.5f}, {ci[1]:.5f}]")
    print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4e}")
    print(f"独立样本数 n = {n}；有水印 gap 高于无水印的条数 {int((diffs > 0).sum())}/{n}")
    print(f"\n详细结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
