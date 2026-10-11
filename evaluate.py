"""
自动评估脚本
计算 PPL（困惑度）和 BLEU（char-level）指标。
采样 val.jsonl 1000 条，输出 evaluate_results.json。
"""
import json
import re
import sys
import time
import random
from pathlib import Path
from collections import defaultdict
import math

import torch

PROJECT_ROOT = Path(__file__).parent
VAL_FILE = PROJECT_ROOT / "data" / "val.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "evaluate_results.json"
SAMPLE_SIZE = 1000
MAX_NEW_TOKENS = 128

# 强制确定性生成
GEN_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
}


def clean_lccc_spaces(text: str) -> str:
    """去除 LCCC 分词空格（与 local_model.py 一致）"""
    text = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', text)
    text = re.sub(r'\s+([，。！？、；：）」』】])', r'\1', text)
    text = re.sub(r'([（「『【])\s+', r'\1', text)
    return text.strip()


def load_samples(n: int) -> list[dict]:
    """从 val.jsonl 随机采样 n 条（行号抽样，避免全量加载）"""
    print(f"加载验证集: {VAL_FILE}")
    # 第一遍：统计总行数（只读换行，很快）
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"  总计 {total} 行")

    # 随机选行号
    random.seed(42)
    indices = set(random.sample(range(total), min(n, total)))

    # 第二遍：只读选中行
    samples = []
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in indices and line.strip():
                samples.append(json.loads(line))
                if len(samples) >= n:
                    break
    print(f"  采样 {len(samples)} 条")
    return samples


def compute_ppl(model, tokenizer, samples: list[dict]) -> dict:
    """逐条计算 PPL"""
    print("\n计算 PPL...")
    losses = []

    for i, item in enumerate(samples):
        msgs = item["messages"]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            continue

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss.item()
            losses.append(loss)

        if (i + 1) % 100 == 0:
            current_ppl = math.exp(sum(losses) / len(losses))
            print(f"  {i+1}/{len(samples)}, PPL={current_ppl:.1f}", flush=True)

    ppls = [math.exp(l) for l in losses]
    ppls.sort()

    results = {
        "mean": round(sum(ppls) / len(ppls), 1),
        "median": round(ppls[len(ppls) // 2], 1),
        "min": round(ppls[0], 1),
        "max": round(ppls[-1], 1),
        "count": len(ppls),
    }
    print(f"  PPL: mean={results['mean']}, median={results['median']}, min={results['min']}, max={results['max']}")
    return results


def char_ngrams(text: str, n: int) -> list[str]:
    """字符级 n-gram"""
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def compute_bleu(reference: str, candidate: str, max_n: int = 4) -> dict:
    """
    简版 char-level BLEU，零依赖。
    参考: BLEU = brevity_penalty * geometric_mean(precisions)
    """
    ref = clean_lccc_spaces(reference)
    cand = clean_lccc_spaces(candidate)

    if not cand or not ref:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    # brevity penalty
    bp = 1.0 if len(cand) >= len(ref) else math.exp(1 - len(ref) / max(len(cand), 1))

    precisions = {}
    for n in range(1, max_n + 1):
        ref_ngrams = defaultdict(int)
        for ng in char_ngrams(ref, n):
            ref_ngrams[ng] += 1

        cand_ngrams = char_ngrams(cand, n)
        if not cand_ngrams:
            precisions[n] = 0.0
            continue

        matches = 0
        for ng in cand_ngrams:
            if ref_ngrams[ng] > 0:
                matches += 1
                ref_ngrams[ng] -= 1

        precisions[n] = matches / len(cand_ngrams)

    # geometric mean, skip zeros
    nonzero = [p for p in precisions.values() if p > 0]
    if not nonzero:
        gm = 0.0
    else:
        gm = math.exp(sum(math.log(p) for p in nonzero) / len(nonzero))

    bleu = bp * gm

    return {
        "bleu1": round(precisions[1] * 100, 1),
        "bleu2": round(precisions[2] * 100, 1),
        "bleu3": round(precisions[3] * 100, 1),
        "bleu4": round(precisions[4] * 100, 1),
    }


def compute_bleu_scores(model, tokenizer, samples: list[dict]) -> dict:
    """计算 char-level BLEU（生成 vs 参考）"""
    print("\n计算 BLEU...", flush=True)
    bleu_sums = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    count = 0
    gen_times = []
    skip_reasons = {"no_pair": 0, "template_err": 0, "empty_reply": 0}

    for i, item in enumerate(samples):
        msgs = item["messages"]
        if len(msgs) < 2:
            continue

        # 取最后一轮 user-assistant 对
        user_msg = None
        ref_msg = None
        for j in range(len(msgs) - 1, 0, -1):
            if msgs[j]["role"] == "assistant" and msgs[j-1]["role"] == "user":
                user_msg = msgs[j-1]["content"]
                ref_msg = msgs[j]["content"]
                break

        if not user_msg or not ref_msg:
            skip_reasons["no_pair"] += 1
            continue

        # 构建推理输入（仅 user，无历史）
        inference_msgs = [{"role": "user", "content": user_msg}]
        try:
            text = tokenizer.apply_chat_template(
                inference_msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:
            skip_reasons["template_err"] += 1
            continue

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.model.generate(**inputs, **GEN_KWARGS, pad_token_id=tokenizer.eos_token_id)
        gen_time = time.time() - t0
        gen_times.append(gen_time)

        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        raw_reply = tokenizer.decode(generated_ids, skip_special_tokens=True)
        gen_reply = clean_lccc_spaces(raw_reply)

        if not gen_reply.strip():
            skip_reasons["empty_reply"] += 1
            continue

        bleu = compute_bleu(ref_msg, gen_reply)
        for k in bleu:
            bleu_sums[k] += bleu[k]
        count += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            avg_b1 = bleu_sums["bleu1"] / count if count else 0
            print(f"  {i+1}/{len(samples)}, BLEU-1={avg_b1:.1f}, valid={count}", flush=True)

    print(f"  跳过: {skip_reasons}", flush=True)

    if count == 0:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "count": 0}

    avg_gen = sum(gen_times) / len(gen_times) if gen_times else 0
    results = {
        "bleu1": round(bleu_sums["bleu1"] / count, 1),
        "bleu2": round(bleu_sums["bleu2"] / count, 1),
        "bleu3": round(bleu_sums["bleu3"] / count, 1),
        "bleu4": round(bleu_sums["bleu4"] / count, 1),
        "count": count,
        "avg_gen_time_ms": round(avg_gen * 1000),
    }
    print(f"  BLEU: {results}")
    return results


def main():
    print("=" * 60)
    print("  Phase 4 自动评估")
    print(f"  验证集: {VAL_FILE}")
    print(f"  采样: {SAMPLE_SIZE} 条")
    print("=" * 60)

    # 延迟导入，避免启动加载模型
    from local_model import get_model

    # 加载模型
    model = get_model()

    # 加载数据
    samples = load_samples(SAMPLE_SIZE)

    # PPL
    ppl_results = compute_ppl(model, model.tokenizer, samples)

    # BLEU（取前 200 条，greedy 生成太慢，1000 条需 ~45 分钟）
    bleu_subset = samples[:200]
    print(f"\nBLEU 评估取 {len(bleu_subset)} 条（减少耗时）", flush=True)
    bleu_results = compute_bleu_scores(model, model.tokenizer, bleu_subset)

    # 汇总输出
    output = {
        "ppl": ppl_results,
        "bleu": bleu_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {OUTPUT_FILE}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
