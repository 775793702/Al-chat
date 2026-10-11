# 中野三玖 AI 对话机器人 — 项目报告

> 日期：2026-06-18 | 作者：自动生成

---

## 1. 项目概述

基于 **Qwen2.5-1.5B-Instruct** 基座模型，使用 **QLoRA**（4-bit NF4量化）微调 LCCC-base 中文对话数据 + 三玖角色风格数据，构建本地中文角色对话机器人。支持双模式运行：本地 QLoRA 模型 / DeepSeek API 回退。

### 技术栈

| 层 | 技术 |
|----|------|
| 基座模型 | Qwen2.5-1.5B-Instruct（1.5B 参数） |
| 微调方法 | QLoRA (4-bit NF4 + LoRA r=8/alpha=32) |
| 训练框架 | transformers Trainer + bitsandbytes + PEFT |
| 情感分析 | SnowNLP（基线） |
| 推理 | TextStreamer 流式 + 后处理 |
| 硬件 | NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) |

---

## 2. 各 Phase 结果

### Phase 1: 数据处理

| 产出文件 | 行数 | 大小 |
|----------|------|------|
| `clean_data.jsonl` | 5,888,376 | 909 MB |
| `miku_style_data.jsonl` | 995 | 376 KB |
| `train.jsonl` | 5,135,662 | 792 MB |
| `val.jsonl` | 570,630 | 88 MB |

- 数据源：LCCC-base + DeepSeek API 生成 995 条三玖角色对话
- 清洗流水线：正则 → jieba分词 → 停用词过滤 → 长度过滤(4-50字) → 去重

### Phase 2: 模型训练

| 指标 | 初始 | 最终 | 改善 |
|------|------|------|------|
| Train Loss | 3.71 (step 100) | 2.17 (step 12500) | -42% |
| Eval Loss | 2.41 (step 500) | 2.14 (step 12500) | - |
| Perplexity | 11.2 | 8.5 | -24% |
| 训练时长 | — | 11.6 小时 | — |
| 显存占用 | — | ~3 GB / 8 GB | — |

- 策略：QLoRA，100K 训练样本，1 epoch，12,500 step
- Adapter 大小：35.3 MB

### Phase 3: 系统集成

| 模块 | 文件 | 功能 |
|------|------|------|
| 本地推理 | `local_model.py` | 4-bit QLoRA 加载 + chat_template + TextStreamer 流式 + LCCC 后处理 |
| 情感识别 | `sentiment.py` | SnowNLP 三分类 (positive/neutral/negative) |
| 配置 | `config.py` | `USE_LOCAL_MODEL` 开关，本地模型参数 |
| 主程序 | `chat.py` | 双模式主循环 + 双轨好感度（关键词+SnowNLP）|

---

## 3. 自动评估

### PPL（困惑度）

| 指标 | 值 |
|------|-----|
| Mean PPL | 64.4 |
| Median PPL | 55.2 |
| Min PPL | 18.3 |
| Max PPL | 281.3 |
| 评估条数 | 1000 |

> PPL 偏高（mean=64.4），中位数 55.2 被少量极端值（max=281.3）拉高。1-epoch 训练导致模型语言建模能力有限。

### BLEU（char-level）

| 指标 | 值 |
|------|-----|
| BLEU-1 | 14.7 |
| BLEU-2 | 4.0 |
| BLEU-3 | 1.1 |
| BLEU-4 | 0.7 |
| 评估条数 | 200 |
| 平均生成时间 | 1127 ms |

> char-level BLEU-1=14.7 对于开放式对话生成属于合理范围。高阶 BLEU 偏低（>2-gram 匹配率骤降）说明生成文本与参考回复在词组层面差异大，符合对话场景多样性预期。

---

## 4. 人工评估

### 评估结论

**人工评估无法有效进行**。模型在加载完整 system prompt（~3000字符 + 世界书条目）后，512 token 上下文窗口被占满，模型无法关注到用户输入，生成为结构化元描述而非对话回复。

### 对比验证

| 条件 | 示例回复 | 结论 |
|------|----------|------|
| 无 system prompt | 「嗯…晚安」「我也是啊」 | 能正常对话，但无角色特征 |
| 有 system prompt | 「### 语调 - 气质上有些不稳…」 | 上下文溢出，生成元描述 |

### 根因
- 训练时 `max_length=512` token，不足以容纳 system prompt + user + assistant
- System prompt 约 3000 字符 → ~1000+ token，远超 512 限制

---

## 5. 对话样例（无 system prompt）

### 样例 1: 日常问候
```
User: 你好呀，今天过得怎么样？
Miku: 我也是啊
```

### 样例 2: 战国话题
```
User: 能给我讲讲日本战国时代的故事吗？
Miku: 嗯如果你开心的话我也很高兴…主要看历史
```

### 样例 3: 道别
```
User: 我要睡觉了，明天再来找你聊天
Miku: 嗯…晚安
```

> 注：以上为无 system prompt 的测试结果。有 system prompt 时生成内容为结构化元描述，不具备对话意义。

---

## 6. 已知问题与改进方向

### 当前局限性

1. **回复质量有限** — 1 epoch 训练（12,500 step），PPL=64.4（mean），模型未充分收敛
   - 原计划 3 epoch，因步速偏慢（~3s/step）改为 1 epoch
2. **上下文窗口严重不足** ⚠️ — `max_length=512` token 无法容纳 system prompt + 用户输入
   - System prompt 约 3000 字符 → ~1000+ token，远超训练窗口
   - 导致有 system prompt 时生成结构化元描述而非对话
   - 无 system prompt 时回复简短但可对话（"晚安""我也是啊"）
3. **回复偏短且偏通用** — 100K 训练样本中三玖角色数据仅 995 条（<1%），角色风格不够强
4. **偶有结构化输出** — 模型生成含 markdown 标题/列表等非对话格式
5. **SnowNLP 精度有限** — 短对话文本上情感误判率较高（~50%）

### 改进方向

| 方向 | 预期效果 | 代价 |
|------|----------|------|
| **增至 3 epoch 训练** | PPL 显著下降 | 训练时间 ×3（~35h）|
| **增大 max_length → 1024/2048** | 容纳完整 system prompt | 训练显存 +30-50% |
| 增大三玖数据占比（1% → 5%） | 角色一致性提升 | 需生成更多角色数据 |
| 换用 Qwen2.5-7B + QLoRA | 生成质量质的飞跃 | 显存不够 |
| 换用更优情感模型（BERT 微调） | 情感准确率提升 | 需标注数据 |
| 精简 system prompt | 快速缓解上下文溢出 | 角色丰富度下降 |

---

## 7. 附录

### 训练超参数

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen/Qwen2.5-1.5B-Instruct |
| 量化 | 4-bit NF4, double quant |
| LoRA r/alpha/dropout | 8/32/0.1 |
| 可训练参数 | 9.2M (0.59%) |
| Batch size (effective) | 8 |
| Learning rate | 2e-4 |
| Scheduler | cosine (warmup 10%) |
| Optimizer | adamw_8bit |
| Epochs | 1 |
| Max length | 512 |

### 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Laptop (8GB VRAM) |
| CPU | Intel (8 cores) |
| RAM | 16 GB |
| OS | Windows 11 Home China |
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 13.1 |

### 复现步骤

```bash
cd D:\Pythonpro\artificial\Al-chat

# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 3. 数据处理（已完成，跳过）
# python data/preprocess.py
# python data/miku_data_generator.py

# 4. 训练（已完成，跳过）
# python train/train.py

# 5. 启动聊天
$env:USE_LOCAL_MODEL = "true"
python chat.py

# 6. 评估
python evaluate.py
python manual_eval.py
```
