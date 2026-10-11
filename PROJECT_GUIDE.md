# PROJECT_GUIDE.md — 简易中文聊天机器人（三玖角色版）

> **给 AI 看的执行指南。按 Phase 顺序执行，每完成一个 Phase 提交一次。**
>
> **项目根目录**：`D:\Pythonpro\artificial\Al-chat`
>
> **参考代码**：`D:\AImiku`（三玖 AI 女友，基于 DeepSeek API 的完整聊天应用）

---

## 执行进度

| Phase | 状态 | 完成时间 | 备注 |
|-------|------|----------|------|
| Phase 0: 环境搭建 | ✅ 完成 | 2026-06-17 | PyTorch 2.11+cu128, RTX 5060 8GB |
| Phase 1: 数据处理 | ✅ 完成 | 2026-06-17 | 5.1M train + 570K val + 995 miku |
| Phase 2: 模型训练 | ✅ 完成 | 2026-06-18 | QLoRA, PPL=8.5, 11.6h |
| Phase 3: 系统集成 | ✅ 完成 | 2026-06-18 | 4 模块，双模式，system prompt 精简版 |
| Phase 4: 评估交付 | ✅ 完成 | 2026-06-18 | PPL=64.4, BLEU-1=14.7, 发现上下文溢出 |
| 报告文档 | ✅ 完成 | 2026-06-18 | AI报告生成提示词.md → 期末设计报告 |

---

## Phase 0：环境搭建 ✅

### 实际结果

- PyTorch: 2.11.0+cu128（CUDA 13.1 驱动兼容）
- CUDA: True, GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- transformers: 5.12.1
- bitsandbytes: 0.49.2（QLoRA 用，Phase 2 新增）
- tensorboard: 2.20.0（训练监控用，Phase 2 新增）
- 从 `D:\AImiku` 复制文件：config.py, miku_character.py, world_book.py, world_book.json ✅
- ⚠️ 注意：需设环境变量 `KMP_DUPLICATE_LIB_OK=TRUE` 解决 PyTorch + Anaconda OpenMP 冲突

---

## Phase 1：数据处理 ✅

### 产出文件

| 文件 | 行数 | 大小 | 说明 |
|------|------|------|------|
| `data/clean_data.jsonl` | 5,888,376 | 909 MB | LCCC 清洗后通用对话 |
| `data/miku_style_data.jsonl` | 995 | 376 KB | 三玖角色风格对话 |
| `data/train.jsonl` | 5,135,662 | 792 MB | 训练集 (90%) |
| `data/val.jsonl` | 570,630 | 88 MB | 验证集 (10%) |

### 实际执行记录

- LCCC 下载方式：直接 HTTP 下载 `silver/lccc`（datasets v5 不兼容旧加载脚本）
- 清洗流水线：正则→jieba分词→停用词过滤(>50%)→长度过滤(4-50字)→去重→格式化
- 三玖数据生成：DeepSeek API, 50 场景×20 条=1000 目标, 实际生成 995 条
- 停用词：哈工大停用词表 767 个
- 去重：训练/验证 0 重叠
- ⚠️ 注意：LCCC 数据含分词空格（silver/lccc 源格式），模型输出也会带空格，Phase 3 需后处理

---

## Phase 2：模型训练 ✅（QLoRA）

> **方案变更**：原设计 bf16 LoRA（显存峰值 ~6-7GB）→ QLoRA 4-bit NF4（显存峰值 ~1.8-3GB），留 5-6GB 余量。

### 方案对比

| | 原方案 (bf16 LoRA) | 最终方案 (QLoRA) |
|------|------|------|
| 基座模型显存 | 2.9 GB | 0.75 GB |
| 峰值显存 | 6-7 GB | 2.5-3 GB |
| 训练时间 | ~8.3h | ~11.6h (+40% 因步速偏慢) |
| 精度损失 | 无 | <0.5% PPL |

### 训练参数（实际）

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | `Qwen/Qwen2.5-1.5B-Instruct` | 缓存至 `models/base_model/`（D盘） |
| 量化 | 4-bit NF4, double quant | bitsandbytes |
| 计算精度 | bf16 | |
| LoRA r/alpha/dropout | 8/32/0.1 | |
| target_modules | q,k,v,o,gate,up,down proj | 9.2M 可训练参数 (0.59%) |
| 训练样本 | 100K train + 10K val | 从 5.1M 中采样 |
| batch_size/grad_accum | 1/8 | 有效 batch=8 |
| epochs | **1**（原计划 3，因步速偏慢改为 1）| |
| max_length | 512 | |
| lr/scheduler/warmup | 2e-4/cosine/10% | |
| optimizer | adamw_8bit | |
| attn_implementation | sdpa | 内存高效注意力 |
| gradient_checkpointing | True | |

### 训练结果

| 指标 | 初始 | 最终 | 改善 |
|------|------|------|------|
| Train Loss | 3.71 (step 100) | **2.17** (step 12500) | ↓42% |
| Eval Loss | 2.41 (step 500) | **2.14** (step 12500) | ↓ |
| Perplexity | 11.2 | **8.5** | ↓24% |
| 训练时长 | — | **11.6 小时** | — |
| 显存占用 | — | ~3 GB / 8 GB | — |

### 训练事件

- CUDA 随机错误 @step 7671（GPU 瞬时故障），从 checkpoint-7500 恢复，损失 ~170 步
- 网络超时 @凌晨 — 改为 `local_files_only=True` 纯本地加载
- Train/Eval loss 全程收敛，无过拟合

### 产出文件

```
models/lora_adapter/
├── adapter_model.safetensors  35.3 MB  ✅
├── adapter_config.json                 ✅
├── tokenizer_config.json               ✅
└── checkpoint-*/                       ✅ (7500, 8000, 8500, ... 12500)

train/
├── train.py                    # QLoRA 训练脚本（250行）
└── train_config.json           # 训练超参数

logs/training/
├── training.log                # 训练日志
└── events.out.tfevents.*       # TensorBoard 事件文件
```

---

## Phase 3：系统集成 ✅

### 目标
将训练好的本地模型接入 AImiku 框架，替换 DeepSeek API，并升级情感识别模块。

### 执行清单

1. ✅ 新建 `local_model.py` — 本地模型加载 + 推理 + 流式输出
2. ✅ 新建 `sentiment.py` — SnowNLP 中文情感识别
3. ✅ 改造 `config.py` — 加入 `USE_LOCAL_MODEL` 开关和本地模型参数
4. ✅ 改造 `chat.py` — 双模式（本地/API）+ 双轨好感度（关键词+SnowNLP）

### 实际产出

| 文件 | 行数 | 功能 |
|------|------|------|
| `local_model.py` | ~270 | 4-bit QLoRA 加载、Qwen chat_template、TextStreamer 流式、LCCC 空格后处理 |
| `sentiment.py` | ~80 | SnowNLP 封装，情绪三分类（positive/neutral/negative），阈值 0.4/0.6 |
| `config.py` | +20 | `USE_LOCAL_MODEL` 开关（环境变量）、本地模型路径/生成参数 |
| `chat.py` | ~490 | 双模式主循环、双轨好感度、`/status` 显示用户情绪和模型模式 |

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| SnowNLP 与好感度关系 | 双轨并行（非替代） | 区分「用户心情」和「对三玖态度」 |
| 流式输出 | TextStreamer 真流式 | token-by-token 打字感，~5行代码 |
| 模型加载时机 | 启动时预加载 | 首次对话即时有回复，不等 10s |
| API 回退 | `USE_LOCAL_MODEL` 环境变量 | 默认 API 模式（向后兼容），`=true` 启用本地 |

### 验证标准
- [x] `python chat.py` 能启动
- [x] 本地模型加载成功（~1s），推理正常（1-3s/条）
- [x] 情感识别正常（positive/neutral/negative）
- [x] `USE_LOCAL_MODEL` 开关可切换本地/API（默认 API 向后兼容）
- [x] 流式输出正常（TextStreamer 逐 token 打印）
- [⚠] 回复质量 — PPL=8.5 (train) / 64.4 (val)，有 system prompt 时上下文溢出
- [x] 本地模式已用精简 system prompt（~200字）临时规避上下文溢出

### 已知问题
- 上下文溢出：训练 `max_length=512` token，完整 system prompt（~1000+ token）会导致模型生成结构化描述而非对话。**临时修复**：本地模式使用精简 system prompt
- 回复质量：1 epoch 训练（12,500 step），无 system prompt 时回复简短但通顺（"晚安""我也是啊"）
- Windows GBK 终端：emoji 和特殊 Unicode 可能乱码，已替换为 ASCII 标签

---

## Phase 4：评估与交付 ✅

### 目标
自动评估（PPL + BLEU）+ 人工评估（10组对话）+ 项目报告。

### 执行清单

1. ✅ 新建 `evaluate.py` — PPL 计算（1000条）+ char-level BLEU（200条）
2. ✅ 新建 `manual_eval.py` — 10 组预设场景 + 交互式评分界面
3. ✅ 撰写 `REPORT.md` — 完整项目报告（7 章节）

### 自动评估结果

| 指标 | 值 | 说明 |
|------|-----|------|
| PPL mean | 64.4 | 1 epoch 训练不足 |
| PPL median | 55.2 | 半数样本在可接受范围 |
| PPL min/max | 18.3 / 281.3 | 极端值拉高均值 |
| BLEU-1 | 14.7 | char-level，对话场景合理 |
| BLEU-2/3/4 | 4.0 / 1.1 / 0.7 | 高阶匹配率低（对话多样性预期内）|
| 生成速度 | 1127 ms/条 | RTX 5060 Laptop 正常 |

### 人工评估

⚠ 人工评估无法有效进行。发现关键问题：**上下文窗口溢出**

| 条件 | 示例回复 | 结论 |
|------|----------|------|
| 无 system prompt | 「嗯…晚安」「我也是啊」 | 能正常对话，简短但通顺 |
| 有 system prompt | 「### 语调 - 气质上有些不稳…」 | 上下文溢出，生成结构化元描述 |

**根因**：训练时 `max_length=512` token，但 system prompt 约 3000 字符 → ~1000+ token，远超训练窗口。模型无法同时关注 system prompt 和 user input。

### 对话样例（无 system prompt）

| 场景 | 用户 | 模型回复 |
|------|------|----------|
| 日常问候 | 你好呀，今天过得怎么样？ | 我也是啊 |
| 战国话题 | 能给我讲讲日本战国时代的故事吗？ | 嗯如果你开心的话我也很高兴…主要看历史 |
| 道别 | 我要睡觉了，明天再来找你聊天 | 嗯…晚安 |

### 产出文件

| 文件 | 功能 |
|------|------|
| `evaluate.py` | 自动评估脚本（PPL + BLEU） |
| `evaluate_results.json` | 评估数值 |
| `manual_eval.py` | 人工评估辅助工具 |
| `manual_eval_gen.jsonl` | 10 组对话生成记录 |
| `REPORT.md` | 完整项目报告 |

### 核心发现

- **上下文溢出是项目当前最大瓶颈**，直接阻碍角色对话功能
- 无 system prompt 时模型可正常对话（证明训练不是完全失败）
- 下一步最优先：**增大 max_length → 1024+，重新训练**

---

## 快速启动

**CMD（命令提示符）**：
```cmd
cd D:\Pythonpro\artificial\Al-chat
set KMP_DUPLICATE_LIB_OK=TRUE

:: API 模式（默认，需 DEEPSEEK_API_KEY）
python chat.py

:: 本地模型模式（精简 system prompt，可正常对话）
set USE_LOCAL_MODEL=true
python chat.py
```

**PowerShell**：
```powershell
cd D:\Pythonpro\artificial\Al-chat
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# API 模式（默认）
python chat.py

# 本地模型模式
$env:USE_LOCAL_MODEL = "true"
python chat.py
```

### 注
- 本地模式启动需等 ~5 秒加载模型
- 本地模式已使用精简 system prompt（~200字），避免 512 token 上下文溢出
- API 模式使用完整 3000 字角色设定，回复质量最佳
- `.env` 文件已从 AImiku 复制，内含 DeepSeek API Key

---

## 关键技术决策记录

| 决策 | 选择 | 原因 | Phase |
|------|------|------|-------|
| 基座模型 | Qwen2.5-1.5B | 中文能力强、轻量、支持 chat_template | 2 |
| 微调方法 | **QLoRA** (4-bit NF4) | 显存峰值仅 ~3GB，安全余量 5GB | 2 |
| 上下文长度 | 512 tokens | 适配 1.5B 模型和 8GB 显存 | 2 |
| ⚠ 上下文长度 | **需增至 1024+** | Phase 4 发现 system prompt 溢出 512 窗口 | 4 |
| 角色数据 | DeepSeek API 批量生成 | 复用 AImiku 资产，质量高 | 1 |
| 训练步数 | Epochs=1（12,500步）| 原计划 3 epoch，步速偏慢 ~3s/step 改为 1 epoch | 2 |
| 模型缓存 | D 盘 models/base_model/ | 不占 C 盘空间 | 2 |
| 情感分析 | SnowNLP（基线）+ 关键词双轨 | 区分「用户心情」和「对三玖态度」| 3 |
| 流式输出 | TextStreamer 真流式 | token-by-token 体验一致 | 3 |
| API 回退 | `USE_LOCAL_MODEL` 开关 | 方便对比本地 vs 云端效果 | 3 |
| 模型加载 | 启动时预加载 | 首次对话即时回复 | 3 |
| BLEU 方法 | char-level 自实现 | 中文不适配词级 BLEU，零依赖 | 4 |

---

## 附录：AImiku 参考代码位置

| 文件 | 路径 | 功能 |
|------|------|------|
| 主聊天程序 | `D:\AImiku\chat.py` | 流式输出、命令系统、好感度、UI |
| 角色设定 | `D:\AImiku\miku_character.py` | System prompt、好感度分级 |
| 配置 | `D:\AImiku\config.py` | API 配置、对话参数 |
| 世界书引擎 | `D:\AImiku\world_book.py` | 关键词匹配、上下文注入 |
| 世界书数据 | `D:\AImiku\world_book.json` | 23条结构化世界观 |
| API Key | `D:\AImiku\.env` | DeepSeek API Key（用于生成训练数据） |

---

> **AI 执行指令**：按 Phase 0 → 1 → 2 → 3 → 4 顺序执行。
> Phase 0-2 已完成。接下来执行 Phase 3。
