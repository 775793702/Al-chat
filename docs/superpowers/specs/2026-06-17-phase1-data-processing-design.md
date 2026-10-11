# Phase 1 数据处理 — 设计文档

> 日期：2026-06-17 | 状态：已确认

## 概述

Phase 1 的目标是为 Qwen2.5-1.5B 本地对话模型准备训练数据。核心工作分三块：LCCC 通用对话数据清洗、DeepSeek API 生成三玖角色风格数据、合并划分训练/验证集。

## 架构

```
data/
├── preprocess.py              # 主控脚本：下载 + 清洗 + 合并 + 划分
├── miku_data_generator.py     # 三玖风格数据生成（独立、断点续跑）
├── stopwords/                 # 哈工大停用词表（自动下载）
├── raw/                       # 原始 LCCC 下载缓存
├── clean_data.jsonl           # 清洗后通用对话 (~10万条)
├── miku_style_data.jsonl      # 三玖风格对话 (~1000条)
├── train.jsonl                # 训练集 (90%)
└── val.jsonl                  # 验证集 (10%)
```

## 数据流

```
HuggingFace LCCC-base
        │
        ▼
  正则清洗（去URL、特殊符号、纯标点行）
        │
        ▼
  jieba 分词 → 统计词数/去停用词占比 → 过滤
        │
        ▼
  长度过滤（4-50字）
        │
        ▼
  格式化为 {"role","content"} 结构
        │
        ▼
  clean_data.jsonl  ──────────┐
                              │
  miku_style_data.jsonl ──────┤  (并行生成)
                              │
        ┌─────────────────────┘
        ▼
  合并 → 随机 shuffle → 9:1 划分
        │
        ▼
  train.jsonl / val.jsonl
```

## 各模块详细设计

### 1. preprocess.py

**下载阶段**：
- 使用 `datasets.load_dataset("lccc", "base")` 下载
- 缓存到 `data/raw/` 避免重复下载

**清洗流水线**：
1. 正则清洗：去掉 URL（`https?://\S+`）、特殊 Unicode 符号、仅含标点的行
2. jieba 分词：对每条文本分词，记录词数和停用词占比
3. 停用词过滤：停用词占比 > 50% 则丢弃整条对话
4. 长度过滤：保留 `user` 和 `assistant` 内容均在 4-50 字之间的对话
5. 格式化为：`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

**停用词表**：
- 来源：https://github.com/goto456/stopwords/blob/master/hit_stopwords.txt
- 自动下载到 `data/stopwords/hit_stopwords.txt`
- 如果下载失败，使用内置的 50 个基础中文停用词兜底

**日志输出**：
- 每步处理后打印剩余条数
- 示例：「正则清洗：120000 → 115000 (去除了 5000 条)」
- 最终打印 5 条样例

### 2. miku_data_generator.py

**输入**：
- API Key：读取 `D:\AImiku\.env` 中的 `DEEPSEEK_API_KEY`
- System Prompt：读取 `D:\AImiku\miku_character.py` 中的 `MIKU_SYSTEM_PROMPT`
- 种子场景：内置 50 个日常对话场景

**生成逻辑**：
- 50 个种子场景 × 每场景 20 条 = 1000 条
- 调用 DeepSeek API（model: deepseek-v4-pro）
- 每条对话格式：`{"messages": [{"role": "user", "content": "种子"}, {"role": "assistant", "content": "三玖回复"}]}`
- 带重试机制（失败重试 3 次，指数退避 2s/4s/8s）
- 断点续跑：每生成 10 条保存一次，重启时检查已有条目数

**种子场景类别**（每类 10 个场景）：
- 日常问候和闲聊
- 分享心情（开心/难过/疲惫）
- 聊战国历史或抹茶（三玖兴奋话题）
- 对方心情不好（触发三玖安慰模式）
- 夸奖或吐槽三玖

**输出**：`data/miku_style_data.jsonl`，每行一个 JSON 对象

### 3. 合并与划分

- 读取 `clean_data.jsonl` + `miku_style_data.jsonl`
- 随机 shuffle（seed=42 保证可复现）
- 9:1 划分为 `train.jsonl` 和 `val.jsonl`
- 打印各文件行数

## 错误处理策略

| 场景 | 处理方式 |
|------|----------|
| LCCC 下载失败（网络） | 重试 3 次，失败则提示手动下载 |
| 停用词表下载失败 | 使用内置 50 词兜底 + 警告 |
| DeepSeek API 调用失败 | 重试 3 次（指数退避），仍失败则记录并跳过 |
| DeepSeek API 限流 | 每批间 sleep 1s，遇到 429 则 sleep 5s |
| 数据量为 0 | 断言失败，终止并提示原因 |

## 验证标准

- [ ] `clean_data.jsonl` 存在且 ≥ 50000 行
- [ ] `miku_style_data.jsonl` 存在且 ≥ 1000 行（允许 5% 失败率，即 ≥ 950）
- [ ] `train.jsonl` 和 `val.jsonl` 存在，比例接近 9:1
- [ ] jieba 分词和去停用词过程在 `preprocess.py` 中有明确日志
- [ ] 清洗后每条数据格式正确（含 `messages` 字段，每条含 `role` 和 `content`）
- [ ] 人工抽检 5 条三玖数据，确认回复符合角色设定

## 技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 脚本拆分 | preprocess.py + miku_data_generator.py | 独立运行、断点续跑、可并行 |
| 分词用于过滤而非替换 | jieba 分词后检查停用词占比，但原文保留 | 训练需要自然文本 |
| 长度范围 | 4-50 字 | LCCC-base 大多在此范围，过短无意义，过长超 512 token |
| API 重试策略 | 3 次指数退避 + 断点续跑 | DeepSeek API 偶尔波动 |
| 随机种子 | seed=42 | 保证数据划分可复现 |
| 三玖数据占比 | ~1%（1000/100000） | 角色风格影响力足够，不破坏通用对话能力 |
