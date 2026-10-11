# Phase 2 模型训练 — 设计文档

> 日期：2026-06-17 | 状态：已确认

## 概述

Phase 2 的目标是用清洗好的对话数据微调 Qwen2.5-1.5B-Instruct，训练一个本地中文对话模型。采用 **QLoRA**（4-bit 量化基座 + LoRA）高效微调，显存峰值仅 ~2GB，adapter 仅 ~50MB。

## 架构

```
train/
├── train.py              # 训练主脚本
└── train_config.json     # 超参数配置

models/
├── base_model/           # Qwen2.5-1.5B 下载缓存（D盘，~3GB）
└── lora_adapter/         # 训练产出：LoRA 权重（~50MB）

logs/
└── training/             # TensorBoard 日志 + training.log
```

## 数据流

```
data/train.jsonl (100K采样) → Tokenize (Qwen chat_template) → DataLoader
                                                                    │
data/val.jsonl   (10K采样)  → Tokenize → DataLoader ──────────────┤
                                                                    │
                                              ┌─────────────────────┘
                                              ▼
                                      Trainer.train()
                                        │  每500 step: eval PPL
                                        │  每1000 step: save checkpoint
                                        ▼
                                  models/lora_adapter/
                                  logs/training/
```

## 关键参数

| 参数 | 值 | 原因 |
|------|-----|------|
| 基座模型 | `Qwen/Qwen2.5-1.5B-Instruct` | 中文强、chat_template 完善 |
| **QLoRA** | 4-bit NF4 量化 + LoRA | 基座模型 2.9GB → 0.75GB |
| bnb_4bit_compute_dtype | bf16 | 计算精度不变 |
| bnb_4bit_quant_type | nf4 | 比 fp4 精度更好 |
| bnb_4bit_use_double_quant | True | 再省 0.1GB |
| LoRA r | 8 | 效果/参数量平衡 |
| LoRA alpha | 32 | α = 4×r |
| LoRA dropout | 0.1 | 防过拟合 |
| target_modules | q,k,v,o,gate,up,down proj | 全覆盖 |
| batch_size | 1 | 适配 8GB VRAM |
| gradient_accumulation | 8 | 有效 batch=8 |
| epochs | 3 | 100K 数据足够收敛 |
| max_length | 512 | 上下文窗口 |
| lr | 2e-4 | LoRA 标准学习率 |
| lr_scheduler | cosine | 平滑衰减 |
| warmup_ratio | 0.1 | 前 10% step 预热 |
| weight_decay | 0.01 | |
| bf16 | True | 计算精度 |
| gradient_checkpointing | True | 省激活值显存 |
| attn_implementation | sdpa | 内存高效注意力 |
| logging_steps | 100 | 每 100 step 记录 |
| eval_steps | 500 | 每 500 step 验证 |
| save_steps | 1000 | checkpoint |
| save_total_limit | 2 | 保留最优+最新 |

## 显存估算（QLoRA）

| 组件 | bf16 LoRA | QLoRA (4-bit) |
|------|-----------|----------------|
| 基座模型 | 2.9 GB | **0.75 GB** |
| LoRA 参数+优化器+梯度 | 57 MB | 57 MB |
| 激活值 (gradient ckpt) | 72 MB | 72 MB |
| Embedding 表 | 445 MB | 445 MB |
| 注意力 (SDPA) | ~20 MB | ~20 MB |
| CUDA 上下文 | 500 MB | 500 MB |
| **理论总计** | 3.9 GB | **~1.8 GB** |
| **实际峰值（含框架开销）** | 6-7 GB | **~2.5-3 GB** |
| RTX 5060 可用 | 8 GB | 8 GB |
| **富裕余量** | 1-2 GB | **5-6 GB** ✅ |

## 训练时间估算

```
100K samples × 3 epochs = 300K effective samples
batch_eff = 1 × 8 = 8
steps = 300K / 8 = 37,500
bf16 LoRA: ~0.8s/step → Total: ~8.3 hours
QLoRA:     ~1.0s/step → Total: ~10 hours  (慢 10-20%)
```

## 兜底方案

| 情况 | 方案 |
|------|------|
| Qwen 下载失败 | 自动降级为 `uer/gpt2-chinese-cluecorpussmall` (~100M params) |
| VRAM 不足 (OOM) | QLoRA 已留 5GB 余量，基本不会发生 |
| 训练中断 | Trainer 自动 resume_from_checkpoint |

## 训练输出

- `models/lora_adapter/adapter_config.json` — LoRA 配置
- `models/lora_adapter/adapter_model.safetensors` — LoRA 权重
- `logs/training/` — TensorBoard 事件文件
- 终端输出 train_loss / eval_loss / eval_perplexity 每 500 step

## 验证标准

- [ ] 训练完成，loss 明显下降（从 ~2.0 到 <1.0）
- [ ] `adapter_model.safetensors` 存在且 ~10-50 MB
- [ ] 验证集 perplexity < 50
- [ ] 能生成通顺中文回复（人工抽检 5 条）
