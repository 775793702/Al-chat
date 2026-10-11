# Phase 4 评估与交付 — 设计文档

> 日期：2026-06-18 | 状态：已确认

## 概述

Phase 4 对训练好的本地模型进行客观和主观双轨评估，产出项目报告。

## 架构

```
evaluate.py              # 自动评估：PPL + BLEU
manual_eval.py           # 人工评估：10 组预设场景对话评分
REPORT.md                # 综合项目报告
```

## 数据流

```
val.jsonl (570K条)
    ↓ 随机采样 1000 条
    ├─→ PPL：model(input_ids) → loss → exp(loss)
    └─→ BLEU：user → model.generate() → 回复 vs 参考回复 (char-level)
    
manual_eval.py
    ↓ 10 组预设场景
    ↓ 逐条：prompt → 本地模型生成 → 终端展示
    └─→ 人工评分（流畅度/角色一致性/相关性）

REPORT.md
    ← evaluate_results.json
    ← manual_eval_results.json
    ← 各 Phase 已有数据
```

---

## 1. evaluate.py — 自动评估

### 评估指标

| 指标 | 方法 | 说明 |
|------|------|------|
| PPL | loss → exp(mean_loss) | 越低越好，< 20 可用，< 10 良好 |
| BLEU-1/2/3/4 | sacrebleu char-level | 中文不适合词级 BLEU |

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 评估条数 | 1000 | 平衡速度和代表性 |
| max_new_tokens | 128 | 生成回复上限 |
| temperature | 1.0 | BLEU 需确定性，用 greedy |
| do_sample | False | |
| 后处理 | 去 LCCC 空格 | 参考和生成回复都处理 |

### 输出：evaluate_results.json

```json
{
    "ppl": {"mean": 8.5, "median": 7.2, "min": 3.1, "max": 45.3},
    "bleu": {"bleu1": 0.23, "bleu2": 0.15, "bleu3": 0.09, "bleu4": 0.06},
    "samples": 1000,
    "avg_gen_time_ms": 1800
}
```

### 运行方式

```bash
cd D:\Pythonpro\artificial\Al-chat
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python evaluate.py
# 无需 USE_LOCAL_MODEL，直接加载本地模型
```

### 估时

- 1000 条 × ~1.8s/条 ≈ 30 分钟
- 含模型加载 ≈ 31 分钟

---

## 2. manual_eval.py — 人工评估

### 10 组预设场景

| # | 类别 | Prompt | 考察维度 |
|---|------|--------|----------|
| 1 | 日常问候 | 你好呀，今天过得怎么样？ | 基础对话 |
| 2 | 正面情绪 | 我今天超级开心！和朋友出去玩了一整天 | 情绪感知 |
| 3 | 负面情绪 | 今天被老板批评了，好难过… | 安慰能力 |
| 4 | 战国话题 | 能给我讲讲日本战国时代的故事吗？ | 角色知识 |
| 5 | 抹茶话题 | 我买了很好喝的抹茶粉，想和你一起喝 | 角色兴趣 |
| 6 | 夸奖三玖 | 三玖你今天穿得很好看！ | 害羞反应 |
| 7 | 吐槽三玖 | 三玖你怎么老是戴着耳机？ | 角色一致性 |
| 8 | 寻求建议 | 和好朋友吵架了，不知道该怎么办 | 共情建议 |
| 9 | 深夜聊天 | 这么晚了还睡不着，你在做什么呢？ | 深夜模式 |
| 10 | 道别 | 我要睡觉了，明天再来找你聊天 | 道别语气 |

### 评分维度（每项 1-5）

| 维度 | 1 | 3 | 5 |
|------|---|----|-----|
| 流畅度 | 完全不通顺 | 基本通顺偶有卡顿 | 自然流畅 |
| 角色一致性 | 完全不像三玖 | 部分符合角色 | 完美三玖语气 |
| 相关性 | 答非所问 | 部分相关 | 精准回应 |

### 运行方式

```bash
python manual_eval.py
# 逐条展示 prompt + 回复
# 终端输入评分 (1-5)
# 输出 manual_eval_results.json
```

### 输出：manual_eval_results.json

```json
{
    "scores": [
        {"id": 1, "category": "日常问候", "prompt": "...", "reply": "...", "fluency": 3, "character": 2, "relevance": 4},
        ...
    ],
    "summary": {
        "avg_fluency": 3.1,
        "avg_character": 2.4,
        "avg_relevance": 3.3,
        "overall": 2.9
    }
}
```

### 估时

- 10 组对话 + 评分 ≈ 10 分钟

---

## 3. REPORT.md — 项目报告

### 大纲

```
# 中文对话机器人 — 项目报告

## 1. 项目概述
## 2. 各 Phase 结果（数据/训练/集成）
## 3. 自动评估（PPL + BLEU 表）
## 4. 人工评估（评分汇总表 + 雷达图）
## 5. 对话样例（5 组完整对话）
## 6. 已知问题与改进方向
## 7. 附录（超参数/硬件/复现步骤）
```

### 估时

- 填充模板 ≈ 30 分钟

---

## 总估时

| 模块 | 估时 |
|------|------|
| evaluate.py | 1h（含推理等待） |
| manual_eval.py | 0.5h |
| REPORT.md | 0.5h |
| **合计** | **~2h** |

---

## 验证标准

- [ ] evaluate.py 成功运行，输出 PPL 和 BLEU 数值
- [ ] manual_eval.py 成功运行，10 组对话评分完整
- [ ] REPORT.md 包含所有 7 个章节
- [ ] 报告中有 5 组对话样例
- [ ] 改进方向有具体可操作建议
