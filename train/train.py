"""
QLoRA 微调 Qwen2.5-1.5B-Instruct 中文对话模型
使用 bitsandbytes 4-bit 量化 + PEFT LoRA + transformers Trainer
"""
import json
import sys
import os
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
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ============================================================
# 路径设置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "train_config.json"
TRAIN_DATA = PROJECT_ROOT / "data" / "train.jsonl"
VAL_DATA = PROJECT_ROOT / "data" / "val.jsonl"

# transformers 5.x: logging_dir 改用环境变量
os.environ["TENSORBOARD_LOGGING_DIR"] = str(PROJECT_ROOT / "logs" / "training")

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
        bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    # 本地文件优先（断网时可用已缓存模型）
    local_only = (Path(cfg["cache_dir"]) / "models--Qwen--Qwen2.5-1.5B-Instruct").exists()

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cfg["cache_dir"],
        trust_remote_code=True,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        local_files_only=local_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        cache_dir=cfg["cache_dir"],
        trust_remote_code=True,
        padding_side="right",
        local_files_only=local_only,
    )

    # 确保 pad_token 设置
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
    将 messages 格式转为模型输入，只对 assistant 回复计算 loss。
    """
    full_texts = []
    prompt_texts = []

    for msgs in examples["messages"]:
        try:
            # 完整对话（含 assistant 回复）
            full_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            full_texts.append(full_text)

            # 仅 prompt（到 assistant header 为止，不含回复）
            # 移除最后一个 assistant 消息，用 add_generation_prompt=True 生成 prompt
            prompt_msgs = [m for m in msgs if m["role"] != "assistant" or m != msgs[-1]]
            # 如果最后一条不是 assistant，回退到去除最后一条
            if len(prompt_msgs) == len(msgs):
                prompt_msgs = msgs[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
            )
            prompt_texts.append(prompt_text)
        except Exception as e:
            logger.warning(f"Failed to apply chat_template, skipping sample: {e}")
            full_texts.append("")
            prompt_texts.append("")

    # Tokenize 完整对话
    tokenized = tokenizer(
        full_texts, truncation=True, max_length=max_length, padding=False,
    )

    # Tokenize prompt（用于确定 mask 位置）
    tokenized_prompts = tokenizer(
        prompt_texts, truncation=True, max_length=max_length, padding=False,
    )

    # 创建 labels：mask 掉 prompt 部分，只保留 assistant 回复
    all_labels = []
    for i, (input_ids, prompt_ids) in enumerate(zip(tokenized["input_ids"], tokenized_prompts["input_ids"])):
        labels = input_ids.copy()
        prompt_len = len(prompt_ids)
        # Mask prompt tokens（含 user 消息 + assistant header）
        labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
        all_labels.append(labels)

    tokenized["labels"] = all_labels
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
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["eval_steps"],  # 与 eval 对齐，确保 load_best_model 可用
        save_total_limit=cfg["save_total_limit"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=int(100000 / 8 * 3 * 0.1),  # =3750 (warmup_ratio replaced in transformers 5.x)
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        optim="adamw_8bit",
        report_to=["tensorboard"],
        # logging_dir removed in transformers 5.x, use env var TENSORBOARD_LOGGING_DIR instead
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # save_safetensors 在 transformers 5.x 默认启用，无需显式指定
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 7. 训练（支持断点续跑）
    output_dir = Path(cfg["output_dir"])
    checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    resume_from_checkpoint = None
    if checkpoints:
        resume_from_checkpoint = str(checkpoints[-1])
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    total_steps_est = len(tokenized_datasets['train']) * cfg['num_train_epochs'] // (cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'])
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"Total steps estimate: {total_steps_est}")
    logger.info("=" * 60)

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 8. 保存
    logger.info("Saving final model...")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # 结果
    elapsed = time.time() - start_time
    logger.info(f"Training complete! Elapsed: {elapsed/3600:.1f} hours")
    logger.info(f"Final train loss: {train_result.training_loss:.4f}")
    logger.info(f"Adapter saved to: {cfg['output_dir']}")

    if train_result.metrics:
        import json
        logger.info(f"Metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in train_result.metrics.items()}, indent=2)}")


if __name__ == "__main__":
    main()
