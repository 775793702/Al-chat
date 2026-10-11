"""
本地模型推理模块
加载 QLoRA 微调后的 Qwen2.5-1.5B-Instruct 模型，提供对话生成接口。

用法:
    model = LocalModel()
    reply = model.chat([
        {"role": "system", "content": "你是三玖..."},
        {"role": "user", "content": "你好呀"}
    ])
"""

import re
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from peft import PeftModel


# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).parent
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = PROJECT_ROOT / "models" / "base_model"
LORA_DIR = PROJECT_ROOT / "models" / "lora_adapter"

# ============================================================
# 默认生成参数
# ============================================================
DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.9,
    "top_p": 0.9,
    "do_sample": True,
}


def _clean_lccc_spaces(text: str) -> str:
    """
    去除 LCCC 训练数据带来的中文分词空格。
    模型训练数据含 jieba 分词空格（如「你好 呀，今天 过 得 怎么 样？」），
    推理时需要还原为自然中文（「你好呀，今天过得怎么样？」）。
    """
    # 中文字符之间的空格（含中日韩统一表意文字）
    text = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', text)
    # 中文标点前多余空格（，。！？、；：）前
    text = re.sub(r'\s+([，。！？、；：）」』】])', r'\1', text)
    # 中文标点后多余空格修复（（「『【 后）
    text = re.sub(r'([（「『【])\s+', r'\1', text)
    # 英文/数字与中文之间的空格保留，只处理纯中文区间
    return text.strip()


class LocalModel:
    """本地 QLoRA 模型封装，提供 OpenAI 兼容的对话接口"""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: Path = CACHE_DIR,
        lora_dir: Path = LORA_DIR,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.lora_dir = lora_dir
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """加载 4-bit 量化基座模型 + LoRA adapter + tokenizer"""
        if self._loaded:
            return

        print("[local_model] 加载模型...", end=" ", flush=True)

        # 1. 量化配置（与训练时一致）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 2. 加载 4-bit 基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )

        # 3. 加载 LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, str(self.lora_dir))
        self.model.eval()

        # 4. 加载 tokenizer（从 adapter 目录，已包含 chat_template）
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.lora_dir),
            trust_remote_code=True,
        )

        # 5. 确保 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._loaded = True
        print("完成")

    @property
    def device(self):
        """底层模型的设备"""
        if self.model is None:
            return torch.device("cpu")
        return self.model.device

    def chat(
        self,
        messages: list[dict],
        stream: bool = True,
        **gen_kwargs,
    ) -> str:
        """
        根据 OpenAI 格式的 messages 生成回复。

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}, ...]
            stream: 是否流式输出到终端
            **gen_kwargs: 覆盖默认生成参数

        Returns:
            生成的回复文本（已后处理）
        """
        if not self._loaded:
            self.load()

        # 1. 应用 chat_template，构建模型输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2. Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        # 3. 设置流式输出
        streamer = None
        if stream:
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            print()  # 换行后开始流式输出

        # 4. 合并生成参数
        kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
        kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        kwargs["streamer"] = streamer

        # 5. 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)

        # 6. 解码（跳过输入部分）
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        raw_reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 7. 后处理：去除 LCCC 分词空格
        reply = _clean_lccc_spaces(raw_reply)

        return reply

    def generate(
        self,
        messages: list[dict],
        **gen_kwargs,
    ) -> str:
        """非流式生成（chat 的 stream=False 别名）"""
        return self.chat(messages, stream=False, **gen_kwargs)


# ============================================================
# 全局单例
# ============================================================

_local_model: LocalModel | None = None


def get_model() -> LocalModel:
    """获取本地模型单例（启动时预加载）"""
    global _local_model
    if _local_model is None:
        _local_model = LocalModel()
        _local_model.load()
    return _local_model


# ============================================================
# 便捷函数
# ============================================================

def chat_once(user_input: str, system_prompt: str = "", stream: bool = True) -> str:
    """
    单轮对话快捷入口。

    Args:
        user_input: 用户输入
        system_prompt: 系统提示（可选）
        stream: 是否流式输出

    Returns:
        模型回复
    """
    model = get_model()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})
    return model.chat(messages, stream=stream)


# ============================================================
# CLI 测试入口
# ============================================================

if __name__ == "__main__":
    """
    快速测试：python local_model.py
    无需启动完整聊天程序，直接测试本地模型是否正常工作。
    """
    print("=" * 50)
    print("  local_model.py 冒烟测试")
    print(f"  模型: {MODEL_NAME}")
    print(f"  LoRA: {LORA_DIR}")
    print("=" * 50)

    model = LocalModel()
    model.load()

    test_inputs = [
        "你好呀，今天过得怎么样？",
        "能给我讲讲日本战国时代的故事吗？",
        "我今天好难过...",
    ]

    for i, user_msg in enumerate(test_inputs):
        print(f"\n{'='*50}")
        print(f"  Test {i+1}/{len(test_inputs)}")
        print(f"{'='*50}")
        print(f"User: {user_msg}")
        print(f"Model: ", end="", flush=True)

        messages = [{"role": "user", "content": user_msg}]
        reply = model.chat(messages, stream=True)

        print()  # 换行
        print(f"  → 回复长度: {len(reply)} 字")
