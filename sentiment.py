"""
中文情感识别模块
基于 SnowNLP 分析用户输入的情绪倾向，用于辅助三玖的回复风格调整。

SnowNLP 情感分数范围 [0, 1]:
  - < 0.4 → negative（负面）
  - 0.4 ~ 0.6 → neutral（中性）
  - > 0.6 → positive（正面）

用法:
    from sentiment import analyze
    result = analyze("我今天好开心！")
    print(result.emotion)   # "positive"
    print(result.score)     # 0.85
    print(result.hint)      # "用户心情很好，可以一起开心地聊天"
"""

from dataclasses import dataclass
from snownlp import SnowNLP


# 阈值
NEGATIVE_THRESHOLD = 0.4
POSITIVE_THRESHOLD = 0.6


@dataclass
class SentimentResult:
    """情感分析结果"""
    emotion: str       # "positive" | "neutral" | "negative"
    score: float       # SnowNLP 原始分值 [0, 1]
    hint: str          # 给三玖的情绪提示文本


def analyze(text: str) -> SentimentResult:
    """
    分析文本情感倾向。

    Args:
        text: 用户输入文本

    Returns:
        SentimentResult 含情绪标签、分值、回复提示
    """
    if not text or not text.strip():
        return SentimentResult(
            emotion="neutral",
            score=0.5,
            hint="",
        )

    s = SnowNLP(text.strip())
    score = s.sentiments

    if score > POSITIVE_THRESHOLD:
        emotion = "positive"
        hint = "对方看起来心情不错，可以一起分享快乐"
    elif score < NEGATIVE_THRESHOLD:
        emotion = "negative"
        hint = "对方似乎情绪低落，需要安慰和陪伴"
    else:
        emotion = "neutral"
        hint = ""

    return SentimentResult(emotion=emotion, score=score, hint=hint)


# ============================================================
# CLI 测试入口
# ============================================================

if __name__ == "__main__":
    """快速测试：python sentiment.py"""
    test_cases = [
        ("我今天好开心！和朋友出去玩了一整天！", "→ positive"),
        ("心情不太好，被老板批评了...", "→ negative"),
        ("今天天气不错", "→ neutral"),
        ("好难过，失恋了...", "→ negative"),
        ("哈哈哈哈超级开心！", "→ positive"),
        ("还行吧，没什么特别的", "→ neutral"),
        ("感觉做什么都不顺心", "→ negative"),
        ("三玖你今天真可爱！", "→ positive"),
    ]

    for text, expected in test_cases:
        result = analyze(text)
        ok = "OK" if result.emotion == expected.split()[-1] else "FAIL"
        print(f"{ok} [{result.emotion:8s}] score={result.score:.3f}  \"{text[:30]}...\"")
        if result.hint:
            print(f"      hint: {result.hint}")
