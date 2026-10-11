# Phase 2 模型训练 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 QLoRA 微调 Qwen2.5-1.5B-Instruct，产出本地对话模型 LoRA adapter。

**Architecture:** 单脚本 `train/train.py` 负责全部流程：采样数据→加载量化基座模型→应用 LoRA→Trainer 训练→保存 adapter。配置与代码分离（`train_config.json`）。

**Tech Stack:** transformers (Trainer), peft (QLoRA), bitsandbytes (4-bit NF4), datasets (streaming load)

**前置条件:** Phase 1 完成（train.jsonl / val.jsonl 存在），bitsandbytes 已安装，KMP_DUPLICATE_LIB_OK=TRUE 环境变量。

**依赖关系:** Task 1 → Task 2 → Task 3 → Task 4 → Task 5

---

### Task 1: 目录准备

**Files:**
- Create: `train/` 目录
- Create: `models/lora_adapter/` 目录
- Create: `logs/training/` 目录

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p "D:\Pythonpro\artificial\Al-chat\train"
mkdir -p "D:\Pythonpro\artificial\Al-chat\models\lora_adapter"
mkdir -p "D:\Pythonpro\artificial\Al-chat\models\base_model"
mkdir -p "D:\Pythonpro\artificial\Al-chat\logs\training"
```

- [ ] **Step 2: 验证**

```bash
ls -la "D:\Pythonpro\artificial\Al-chat\train/"
ls -la "D:\Pythonpro\artificial\Al-chat\models/"
ls -la "D:\Pythonpro\artificial\Al-chat\logs/"
```

---

### Task 2: 训练配置文件

**Files:**
- Create: `train/train_config.json`

- [ ] **Step 1: 创建 train_config.json**

```json
{
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "cache_dir": "D:\\Pythonpro\\artificial\\Al-chat\\models\\base_model",

    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,

    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    "train_sample_size": 100000,
    "val_sample_size": 10000,
    "max_length": 512,

    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,

    "bf16": true,
    "gradient_checkpointing": true,
    "attn_implementation": "sdpa",

    "logging_steps": 100,
    "eval_steps": 500,
    "save_steps": 1000,
    "save_total_limit": 2,

    "output_dir": "D:\\Pythonpro\\artificial\\Al-chat\\models\\lora_adapter",
    "logging_dir": "D:\\Pythonpro\\artificial\\Al-chat\\logs\\training"
}
```

- [ ] **Step 2: 验证 JSON 格式**

```bash
python -c "import json; json.load(open('D:\\Pythonpro\\artificial\\Al-chat\\train\\train_config.json')); print('JSON valid')"
```

---

### Task 3: 训练脚本

**Files:**
- Create: `train/train.py`

- [ ] **Step 1: 创建训练脚本** — 完整写入以下代码

```python
"""
QLoRA 微调 Qwen2.5-1.5B-Instruct 中文对话模型
使用 bitsandbytes 4-bit 量化 + PEFT LoRA + transformers Trainer
"""
import json
import sys
import time
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# ============================================================
# 路径设置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "train_config.json"
TRAIN_DATA = PROJECT_ROOT / "data" / "train.jsonl"
VAL_DATA = PROJECT_ROOT / "data" / "val.jsonl"

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "training" / "training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

set_seed(42)

# ============================================================
# 加载配置
# ============================================================
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

logger.info(f"Config loaded: model={cfg['model_name']}")
logger.info(f"Output: {cfg['output_dir']}")
logger.info(f"Sample sizes: train={cfg['train_sample_size']}, val={cfg['val_sample_size']}")


# ============================================================
# 采样并加载数据
# ============================================================
def load_and_sample_data(train_path, val_path, train_size, val_size):
    """从 jsonl 文件采样数据，返回 HuggingFace Dataset"""
    logger.info("Loading and sampling data...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
        split={
            "train": f"train[:{train_size}]",
            "validation": f"validation[:{val_size}]",
        },
    )
    logger.info(f"Train: {len(dataset['train'])} samples, Val: {len(dataset['validation'])} samples")
    return dataset


# ============================================================
# 加载 QLoRA 模型
# ============================================================
def load_qlora_model():
    """加载 4-bit 量化基座模型 + tokenizer"""
    logger.info(f"Loading model: {cfg['model_name']}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cfg["cache_dir"],
        trust_remote_code=True,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        cache_dir=cfg["cache_dir"],
        trust_remote_code=True,
        padding_side="right",
    )

    # Qwen tokenizer 有 pad_token，但确保设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备 kbit 训练
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # 训练时关闭 KV cache

    # 应用 LoRA
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# Tokenize 数据
# ============================================================
def tokenize_function(examples, tokenizer, max_length):
    """
    将 messages 格式转为模型输入：
    [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
    """
    texts = []
    for msgs in examples["messages"]:
        # 使用 chat_template 生成训练文本
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        except Exception:
            texts.append("")

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # DataCollator 会动态 padding
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ============================================================
# 主训练流程
# ============================================================
def main():
    start_time = time.time()

    # 1. 加载数据
    dataset = load_and_sample_data(
        TRAIN_DATA, VAL_DATA,
        cfg["train_sample_size"], cfg["val_sample_size"],
    )

    # 2. 加载模型
    model, tokenizer = load_qlora_model()

    # 3. Tokenize
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, cfg["max_length"]),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # 4. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        logging_dir=cfg["logging_dir"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        optim="adamw_8bit",  # 8-bit AdamW 进一步省显存
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        seed=42,
        dataloader_num_workers=0,  # Windows 下多进程 DataLoader 容易出问题
        remove_unused_columns=False,
        group_by_length=False,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7. 训练
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"Total steps estimate: {len(tokenized_datasets['train']) * cfg['num_train_epochs'] // (cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'])}")
    logger.info("=" * 60)

    train_result = trainer.train()

    # 8. 保存
    logger.info("Saving final model...")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # 打印结果
    elapsed = time.time() - start_time
    logger.info(f"Training complete! Elapsed: {elapsed/3600:.1f} hours")
    logger.info(f"Final train loss: {train_result.training_loss:.4f}")
    logger.info(f"Adapter saved to: {cfg['output_dir']}")

    # 输出训练指标
    if train_result.metrics:
        logger.info(f"Metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in train_result.metrics.items()}, indent=2)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本语法**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "import py_compile; py_compile.compile('train/train.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: 快速冒烟测试（验证模型加载+tokenize，不训练）**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "
import json, torch, sys
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load config
with open('train/train_config.json', 'r') as f:
    cfg = json.load(f)

print('1/4 Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    cfg['model_name'],
    cache_dir=cfg['cache_dir'],
    trust_remote_code=True,
)

# Test chat template
msgs = [{'role':'user','content':'你好'},{'role':'assistant','content':'你好呀'}]
text = tokenizer.apply_chat_template(msgs, tokenize=False)
print(f'   chat_template OK: {text[:60]}...')

print('2/4 Loading QLoRA model (this will download ~3GB on first run)...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    cfg['model_name'],
    quantization_config=bnb_config,
    device_map='auto',
    cache_dir=cfg['cache_dir'],
    trust_remote_code=True,
    attn_implementation='sdpa',
)
print('   Model loaded')

print('3/4 Applying LoRA...')
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=cfg['lora_target_modules'], bias='none', task_type='CAUSAL_LM')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print('4/4 Test forward pass...')
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
print(f'   Forward pass OK, loss={outputs.loss.item():.4f}')
print('ALL SMOKE TESTS PASSED!')
" 2>&1 | head -30
```

Expected: chat_template OK, model loaded, LoRA applied, forward pass OK.

> ⚠ 首次运行会从 HuggingFace 下载 Qwen2.5-1.5B-Instruct（~3GB）到 `models/base_model/`。预计 5-15 分钟。

---

### Task 4: 执行训练

**Files:** 无新建，产出 `models/lora_adapter/` 和 `logs/training/`

- [ ] **Step 1: 启动训练（后台执行）**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && nohup python train/train.py > logs/training/train_stdout.log 2>&1 &
echo "Training started in background, PID: $!"
```

> ⚠ 训练预计 ~10 小时。可通过 `tensorboard --logdir logs/training/` 监控进度。
>
> 训练过程中会自动：
> - 每 100 step 记录 loss 到 tensorboard 和日志
> - 每 500 step 在验证集上评估 PPL
> - 每 1000 step 保存 checkpoint
> - 训练结束后加载最优 checkpoint 并保存到 `models/lora_adapter/`

- [ ] **Step 2: 监控训练进度**

```bash
# 查看最近日志
tail -20 "D:\Pythonpro\artificial\Al-chat\logs\training\training.log"

# 查看 GPU 使用
nvidia-smi

# 查看已保存的 checkpoint
ls -la "D:\Pythonpro\artificial\Al-chat\models\lora_adapter/"
```

---

### Task 5: 验证

**Files:** 无新建

- [ ] **Step 1: 检查产出文件**

```bash
python -c "
from pathlib import Path
adapter_dir = Path('D:/Pythonpro/artificial/Al-chat/models/lora_adapter')
files = ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer_config.json']
for fn in files:
    p = adapter_dir / fn
    exists = p.exists()
    size_mb = p.stat().st_size / 1024**2 if exists else 0
    print(f'{\"OK\" if exists else \"MISSING\"} {fn}: {size_mb:.1f} MB')
"
```
Expected: 所有文件存在，`adapter_model.safetensors` ~20-50 MB

- [ ] **Step 2: 验证训练日志**

```bash
python -c "
import re
with open('D:/Pythonpro/artificial/Al-chat/logs/training/training.log', 'r', encoding='utf-8') as f:
    log = f.read()
    # 找到 loss
    losses = re.findall(r'loss[:\s=]+([\d.]+)', log)
    if losses:
        print(f'Train loss entries found: {len(losses)}')
        print(f'  Initial loss: {losses[0]}')
        print(f'  Final loss: {losses[-1]}')
        # Check loss decreased
        if float(losses[-1]) < float(losses[0]):
            print('  [OK] Loss decreased')
        else:
            print('  [WARN] Loss did not decrease')
    # 找到 perplexity
    ppls = re.findall(r'perplexity[:\s=]+([\d.]+)', log)
    if ppls:
        print(f'Perplexity entries found: {len(ppls)}')
        print(f'  Best PPL: {min(float(p) for p in ppls)}')
        if min(float(p) for p in ppls) < 50:
            print('  [OK] PPL < 50')
        else:
            print(f'  [WARN] PPL >= 50: {min(float(p) for p in ppls)}')
    else:
        print('Check eval loss:')
        eval_losses = re.findall(r'eval_loss[:\s=]+([\d.]+)', log)
        if eval_losses:
            print(f'  Eval losses: {eval_losses[:5]}...')
"
```
Expected: Loss 下降趋势，PPL < 50（或 eval_loss 持续下降）

- [ ] **Step 3: 生成测试 — 人工抽检 5 条回复**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "
import json, torch, sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

# Load config
with open('train/train_config.json', 'r') as f:
    cfg = json.load(f)

# Load base model (4-bit)
import torch
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg['model_name'], quantization_config=bnb_config, device_map='auto',
    cache_dir=cfg['cache_dir'], trust_remote_code=True, attn_implementation='sdpa',
)
model = PeftModel.from_pretrained(model, cfg['output_dir'])
tokenizer = AutoTokenizer.from_pretrained(cfg['output_dir'], trust_remote_code=True)

# Test prompts
test_prompts = [
    '你好呀，今天过得怎么样？',
    '我今天好难过，被朋友误会了',
    '能给我讲讲日本战国时代的故事吗？',
    '你最喜欢什么口味的抹茶？',
    '哈哈哈哈今天超级开心！',
]

for i, prompt in enumerate(test_prompts):
    msgs = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.9, do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    
    reply = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f'\\n=== Test {i+1} ===')
    print(f'User: {prompt}')
    print(f'Model: {reply.strip()[:200]}')
"
```
Expected: 5 条回复通顺、有意义、与输入相关。

- [ ] **Step 4: 最终检查清单**

- [ ] `models/lora_adapter/adapter_model.safetensors` 存在（~20-50 MB）
- [ ] `models/lora_adapter/adapter_config.json` 存在
- [ ] 训练日志中 loss 明显下降
- [ ] 验证集 PPL < 50（或 eval_loss 持续下降）
- [ ] 5 条生成回复通顺、有意义
- [ ] 训练时间在预期范围内（~10h）
