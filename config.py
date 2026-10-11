"""
DeepSeek API 配置模块
加载环境变量，初始化 OpenAI 兼容客户端
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# === API 配置 ===
# DeepSeek V4 API: https://api.deepseek.com
# 可用模型: deepseek-v4-pro (1.6T/49B), deepseek-v4-flash (284B/13B)
# 旧模型 deepseek-chat/deepseek-reasoner 将于 2026-07-24 弃用
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# === 对话参数 ===
# DeepSeek 推荐: temperature=1.0, top_p=1.0
TEMPERATURE = 1.0        # 角色扮演适合偏高温度
MAX_TOKENS = 1024        # 单次回复最大长度
TOP_P = 1.0              # nucleus sampling
FREQUENCY_PENALTY = 0.0  # DeepSeek 推荐不使用
PRESENCE_PENALTY = 0.0   # DeepSeek 推荐不使用

# === 记忆设置 ===
MAX_HISTORY_TURNS = 50   # 保留最近50轮对话作为上下文
AUTO_SAVE = True         # 自动保存对话

# === 当前会话 ===
CONVERSATION_FILE = Path(__file__).parent / "conversation.json"


def create_client() -> OpenAI:
    """创建 DeepSeek API 客户端（OpenAI 兼容）"""
    if not API_KEY or API_KEY == "sk-your-api-key-here":
        raise ValueError(
            "❌ 请先在 .env 文件中设置你的 DeepSeek API Key！\n"
            "   打开 D:\\AImiku\\.env，将 DEEPSEEK_API_KEY 替换为你的真实 Key。\n"
            "   获取 Key: https://platform.deepseek.com/api_keys"
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_chat_params() -> dict:
    """返回对话生成参数"""
    return {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "stream": True,
    }


# ============================================================
# 本地模型配置（Phase 3 新增）
# ============================================================

# 是否使用本地模型（False = DeepSeek API, True = 本地 QLoRA 模型）
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() in ("true", "1", "yes")

# 本地模型路径
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_CACHE_DIR = Path(__file__).parent / "models" / "base_model"
LOCAL_LORA_DIR = Path(__file__).parent / "models" / "lora_adapter"

# 本地模型生成参数
LOCAL_MAX_NEW_TOKENS = 256
LOCAL_TEMPERATURE = 0.9
LOCAL_TOP_P = 0.9
