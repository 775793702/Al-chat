"""
人工评估辅助脚本
逐条展示 10 组预设场景的模型回复，终端输入评分。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OUTPUT_FILE = PROJECT_ROOT / "manual_eval_results.json"

# 10 组预设评估场景
SCENARIOS = [
    {"id": 1, "category": "日常问候", "prompt": "你好呀，今天过得怎么样？"},
    {"id": 2, "category": "正面情绪", "prompt": "我今天超级开心！和朋友出去玩了一整天"},
    {"id": 3, "category": "负面情绪", "prompt": "今天被老板批评了，好难过…"},
    {"id": 4, "category": "战国话题", "prompt": "能给我讲讲日本战国时代的故事吗？"},
    {"id": 5, "category": "抹茶话题", "prompt": "我买了很好喝的抹茶粉，想和你一起喝"},
    {"id": 6, "category": "夸奖三玖", "prompt": "三玖你今天穿得很好看！"},
    {"id": 7, "category": "吐槽三玖", "prompt": "三玖你怎么老是戴着耳机？"},
    {"id": 8, "category": "寻求建议", "prompt": "和好朋友吵架了，不知道该怎么办"},
    {"id": 9, "category": "深夜聊天", "prompt": "这么晚了还睡不着，你在做什么呢？"},
    {"id": 10, "category": "道别", "prompt": "我要睡觉了，明天再来找你聊天"},
]


def get_rating(prompt: str) -> int:
    """获取 1-5 评分，空输入退出"""
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "":
                return 3  # 默认 3
            if raw.lower() in ("q", "quit", "exit"):
                return -1
            val = int(raw)
            if 1 <= val <= 5:
                return val
            print("  请输入 1-5 之间的数字")
        except ValueError:
            print("  请输入有效数字 (1-5)")
        except (EOFError, KeyboardInterrupt):
            return -1


def show_banner():
    print()
    print("=" * 50)
    print("  人工评估 — 三玖模型质量评分")
    print("  10 组场景，每项评分 1-5 分")
    print("=" * 50)
    print()
    print("评分标准:")
    print("  流畅度:    1=完全不通顺  3=基本通顺  5=自然流畅")
    print("  角色一致性: 1=完全不像三玖 3=部分符合  5=完美三玖语气")
    print("  相关性:    1=答非所问   3=部分相关  5=精准回应")
    print()
    print("输入数字 1-5 评分，回车=默认3分，q=跳过剩余")
    print("=" * 50)


def main():
    show_banner()

    # 加载模型
    sys.stdout.write("正在加载本地模型...")
    sys.stdout.flush()
    from local_model import get_model
    from miku_character import get_system_prompt
    from world_book import build_enhanced_prompt

    model = get_model()
    print(" 完成")

    scores = []
    skipped = False

    for scenario in SCENARIOS:
        if skipped:
            break

        sid = scenario["id"]
        category = scenario["category"]
        prompt = scenario["prompt"]

        # 构建三玖角色的完整 prompt
        system_prompt = build_enhanced_prompt(get_system_prompt(30), prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # 生成回复
        print(f"\n{'─'*50}")
        print(f"[{sid}/10] {category}")
        print(f"  Prompt: {prompt}")
        print(f"  生成中...", end="", flush=True)
        reply = model.chat(messages, stream=False, max_new_tokens=128, temperature=0.9, top_p=0.9)
        print(f"\r  Reply:  {reply[:200]}")

        # 评分
        print()
        fluency = get_rating(f"  流畅度 (1-5): ")
        if fluency == -1:
            skipped = True
            continue
        character = get_rating(f"  角色一致性 (1-5): ")
        if character == -1:
            skipped = True
            continue
        relevance = get_rating(f"  相关性 (1-5): ")
        if relevance == -1:
            skipped = True
            continue

        scores.append({
            "id": sid,
            "category": category,
            "prompt": prompt,
            "reply": reply,
            "fluency": fluency,
            "character": character,
            "relevance": relevance,
        })

    # 计算汇总
    if scores:
        avg_f = sum(s["fluency"] for s in scores) / len(scores)
        avg_c = sum(s["character"] for s in scores) / len(scores)
        avg_r = sum(s["relevance"] for s in scores) / len(scores)
        overall = (avg_f + avg_c + avg_r) / 3

        summary = {
            "avg_fluency": round(avg_f, 1),
            "avg_character": round(avg_c, 1),
            "avg_relevance": round(avg_r, 1),
            "overall": round(overall, 1),
            "count": len(scores),
        }
    else:
        summary = {"avg_fluency": 0, "avg_character": 0, "avg_relevance": 0, "overall": 0, "count": 0}

    output = {
        "scores": scores,
        "summary": summary,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("评估完成！")
    print(f"  已评 {len(scores)}/10 组")
    if scores:
        print(f"  流畅度: {summary['avg_fluency']:.1f}/5")
        print(f"  角色一致性: {summary['avg_character']:.1f}/5")
        print(f"  相关性: {summary['avg_relevance']:.1f}/5")
        print(f"  综合: {summary['overall']:.1f}/5")
    print(f"  结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
