# -*- coding: utf-8 -*-
"""
run_experiment_synthid.py —— SynthID-Text 配对机器翻译实验（本地模型版）

设计：同一段落、同一随机种子，分别在「水印关闭」与「水印开启」两种条件下各生成一次，
      形成严格配对样本，供后续水印可检测性 / 译文质量对照分析。

用法：
    python scripts/run_experiment_synthid.py                # 全部 60 段 × 2 条件
    python scripts/run_experiment_synthid.py --limit 3      # 冒烟测试
    python scripts/run_experiment_synthid.py --4bit         # 显存不足时 4-bit 量化加载
    python scripts/run_experiment_synthid.py --in data/translation_results.csv --out data/regenerated.csv

输入约定：CSV 需含中文原文列（`zh_source` 或 `段落原文`）与编号列（`id` 或 `编号`）；
          直接指向随仓库发布的 data/translation_results.csv 即可从源文重跑。
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
)

# ============================================================
# 1. 路径与实验配置
# ============================================================
BASE = Path(__file__).resolve().parent          # scripts/ 目录
ROOT = BASE.parent                               # 仓库根
DATA = ROOT / "data"

CSV_PATH = DATA / "translation_results.csv"
OUTPUT_PATH = DATA / "translation_results_generated.csv"

MODEL_ID = Path(os.environ.get(
    "QWEN_MODEL_DIR", str(ROOT.parent / "experiment" / "Qwen2.5-3B-Instruct")))

SEED = 42

# 采样参数：两种条件必须完全一致，否则配对失效
GEN_KWARGS = dict(
    max_new_tokens=1024,
    temperature=1.0,
    do_sample=True,
    top_p=0.95,
)

SYNTHID_CONFIG = SynthIDTextWatermarkingConfig(
    keys=[
        654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
        590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
    ],
    ngram_len=5,
    sampling_table_size=2**16,
    sampling_table_seed=0,
    context_history_size=1024,
)


# ============================================================
# 2. 加载模型与分词器
# ============================================================
def load_model(use_4bit: bool):
    print(f"正在从本地加载模型 {MODEL_ID} ...", flush=True)
    if not MODEL_ID.exists():
        raise FileNotFoundError(
            f"本地模型目录不存在：{MODEL_ID}\n"
            "请先下载 Qwen2.5-3B-Instruct，并设置环境变量 QWEN_MODEL_DIR 指向该目录。"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID), local_files_only=True)

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ID), **kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"加载完成｜dtype={model.dtype}｜设备={model.device}", flush=True)
    return tokenizer, model


# ============================================================
# 3. 翻译函数
# ============================================================
def build_prompt(tokenizer, zh_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a highly accurate, professional Chinese-to-English translator.",
        },
        {
            "role": "user",
            "content": (
                "Please translate the following Chinese expository text into fluent, natural, "
                "and academically appropriate English.\n"
                "Constraints:\n"
                "1. Do NOT add any commentary, explanations, or extraneous text.\n"
                "2. Do NOT summarize or omit any information from the source.\n"
                "3. Maintain the original tone and logical structure of the expository text.\n\n"
                f"Chinese Text:\n{zh_text}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def translate(tokenizer, model, prompt: str, watermarking_config=None):
    """执行一次翻译，返回 (译文, 生成 token 数)。"""
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # 每次生成前重置种子：保证两种条件从同一个采样起点出发
    # （注意：这只能保证起点一致，生成本身仍是随机采样，故两版译文必然不同）
    torch.manual_seed(SEED)

    generate_kwargs = dict(GEN_KWARGS)
    if watermarking_config is not None:
        generate_kwargs["watermarking_config"] = watermarking_config

    with torch.no_grad():
        outputs = model.generate(
            **inputs, **generate_kwargs, pad_token_id=tokenizer.pad_token_id
        )

    input_len = inputs.input_ids.shape[1]
    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, int(gen_ids.shape[0])


# ============================================================
# 4. 主循环
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(CSV_PATH), help="输入 CSV（含中文原文列）")
    ap.add_argument("--out", dest="out_path", default=str(OUTPUT_PATH), help="输出 CSV 路径")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 段（0 表示全部）")
    ap.add_argument("--4bit", dest="use_4bit", action="store_true", help="4-bit 量化加载")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    id_col = "id" if "id" in df.columns else "编号"
    text_col = "zh_source" if "zh_source" in df.columns else "段落原文"
    if args.limit:
        df = df.head(args.limit)
    total = len(df)

    # 断点续跑：已有结果的编号跳过
    done = set()
    out_path = Path(args.out_path)
    if out_path.exists():
        try:
            done = set(pd.read_csv(out_path)[id_col].astype(str))
            print(f"检测到已有结果 {len(done)} 条，将跳过这些段落", flush=True)
        except Exception:
            done = set()

    tokenizer, model = load_model(args.use_4bit)
    results = []
    if out_path.exists() and done:
        results = pd.read_csv(out_path).to_dict("records")

    t_start = time.time()
    for i, row in df.iterrows():
        doc_id = str(row[id_col])
        if doc_id in done:
            continue
        zh_text = row[text_col]
        prompt = build_prompt(tokenizer, zh_text)

        print(f"[{i + 1}/{total}] {doc_id} - 无水印版本 ...", flush=True)
        t0 = time.time()
        en_off, n_off = translate(tokenizer, model, prompt, None)
        t_off = time.time() - t0

        print(f"[{i + 1}/{total}] {doc_id} - SynthID-Text 版本 ...", flush=True)
        t0 = time.time()
        en_on, n_on = translate(tokenizer, model, prompt, SYNTHID_CONFIG)
        t_on = time.time() - t0

        results.append({
            "id": doc_id,
            "group": row.get("group", row.get("组别", "")),
            "topic": row.get("topic", row.get("学科主题", "")),
            "zh_source": zh_text,
            "en_watermark_off": en_off,
            "en_watermark_on": en_on,
            "sec_off": round(t_off, 1),
            "sec_on": round(t_on, 1),
            "tokens_off": n_off,
            "tokens_on": n_on,
            "seed": SEED,
        })
        elapsed = time.time() - t_start
        remain = (total - (i + 1)) * (t_off + t_on)
        print(f"    off {n_off} tok / {t_off:.1f}s｜on {n_on} tok / {t_on:.1f}s"
              f"｜已用 {elapsed/60:.1f} min，预计剩余 {remain/60:.1f} min", flush=True)

        # 每 5 篇落盘一次，防止中断丢结果
        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全部完成｜{len(results)} 条结果已保存至 {out_path}")
    print(f"总耗时 {(time.time() - t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
