"""
中野三玖 AI 女友 — 终端聊天程序
支持 DeepSeek API / 本地 QLoRA 模型双模式
带流式输出、对话记忆、好感度系统、情感识别
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from colorama import init, Fore, Style
from openai import OpenAI

from config import (
    create_client, get_chat_params, CONVERSATION_FILE,
    MAX_HISTORY_TURNS, AUTO_SAVE,
    USE_LOCAL_MODEL, LOCAL_MAX_NEW_TOKENS, LOCAL_TEMPERATURE, LOCAL_TOP_P,
)
from miku_character import get_system_prompt
from world_book import build_enhanced_prompt
from sentiment import analyze as analyze_sentiment

# 初始化 colorama (Windows 彩色终端)
init(autoreset=True)

# === 颜色定义 ===
MIKU_COLOR = Fore.LIGHTCYAN_EX     # 三玖的回复
USER_COLOR = Fore.LIGHTGREEN_EX    # 用户的输入
SYSTEM_COLOR = Fore.LIGHTYELLOW_EX # 系统提示
ERROR_COLOR = Fore.LIGHTRED_EX     # 错误信息
DIM_COLOR = Fore.LIGHTBLACK_EX     # 次要信息
HIGHLIGHT = Fore.LIGHTMAGENTA_EX   # 高亮

MIKU_NAME = f"{MIKU_COLOR}三玖{Style.RESET_ALL}"
USER_NAME = f"{USER_COLOR}你{Style.RESET_ALL}"


# === 好感度关键词 ===
POSITIVE_KEYWORDS = [
    "三玖", "可爱", "喜欢", "温柔", "厉害", "谢谢", "miku",
    "想你", "想你了", "抱抱", "亲亲", "么么", "晚安", "早安",
    "好看", "漂亮", "聪明", "有趣", "开心", "最喜欢", "miku",
    "一緒", "すごい", "かわいい",
]
NEGATIVE_KEYWORDS = [
    "烦", "无聊", "滚", "走开", "讨厌", "别烦我", "不想理",
    "闭嘴", "别说了",
]


# ============================================================
# 对话状态管理
# ============================================================

class ConversationState:
    """管理对话状态：消息历史、好感度、心情、用户情绪"""

    def __init__(self):
        self.messages: list[dict] = []       # API 兼容的消息列表
        self.total_turns: int = 0            # 总对话轮次
        self.affection: int = 0              # 好感度 0-100
        self.mood: str = "neutral"           # 三玖当前心情
        self.user_emotion: str = "neutral"   # 用户当前情绪 (SnowNLP)
        self.created_at: str = ""            # 创建时间
        self.last_saved: str = ""            # 最后保存时间

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "total_turns": self.total_turns,
            "affection": self.affection,
            "mood": self.mood,
            "user_emotion": self.user_emotion,
            "created_at": self.created_at,
            "last_saved": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        state = cls()
        state.messages = data.get("messages", [])
        state.total_turns = data.get("total_turns", 0)
        state.affection = min(data.get("affection", 0), 100)
        state.mood = data.get("mood", "neutral")
        state.user_emotion = data.get("user_emotion", "neutral")
        state.created_at = data.get("created_at", "")
        return state


# ============================================================
# 持久化
# ============================================================

def load_conversation() -> ConversationState:
    """从文件加载对话历史"""
    if CONVERSATION_FILE.exists():
        try:
            with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = ConversationState.from_dict(data)
            return state
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{ERROR_COLOR}⚠ 对话文件损坏，将开始新对话: {e}")
    return ConversationState()


def save_conversation(state: ConversationState):
    """保存对话到文件"""
    try:
        data = state.to_dict()
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"{ERROR_COLOR}⚠ 保存失败: {e}")


# ============================================================
# 好感度系统
# ============================================================

def update_affection(state: ConversationState, user_input: str):
    """
    双轨好感度更新：
      轨道 1 - 关键词匹配：检测用户对三玖的态度（正面/负面词）
      轨道 2 - SnowNLP 情感分析：检测用户自身情绪状态
    """
    text_lower = user_input.lower()

    # === 轨道 1: 关键词匹配（对三玖的态度） ===
    added = 0
    for kw in POSITIVE_KEYWORDS:
        if kw in text_lower or kw in user_input:
            added += 2
            break

    for kw in NEGATIVE_KEYWORDS:
        if kw in text_lower or kw in user_input:
            added -= 3
            break

    if added == 0:
        added = 1  # 基础交互加分

    state.affection = max(0, min(100, state.affection + added))

    # === 轨道 2: SnowNLP 情绪检测（用户自身状态）===
    sentiment_result = analyze_sentiment(user_input)
    state.user_emotion = sentiment_result.emotion

    # === 更新三玖心情 ===
    if state.affection >= 70:
        state.mood = "warm"
    elif state.affection >= 40:
        state.mood = "happy"
    elif state.affection >= 15:
        state.mood = "neutral"
    else:
        state.mood = "shy"


def get_mood_emoji(mood: str) -> str:
    """心情对应的 emoji"""
    return {
        "shy": "😳",
        "neutral": "😶",
        "happy": "😊",
        "warm": "💕",
    }.get(mood, "😶")


def get_affection_label(affection: int) -> str:
    """好感度标签"""
    if affection >= 80:
        return f"{HIGHLIGHT}♥ 恋心{Style.RESET_ALL}"
    elif affection >= 50:
        return f"{Fore.LIGHTMAGENTA_EX}♡ 亲密{Style.RESET_ALL}"
    elif affection >= 25:
        return f"{Fore.LIGHTCYAN_EX}☆ 友好{Style.RESET_ALL}"
    elif affection >= 10:
        return f"{Fore.LIGHTBLUE_EX}・ 初识{Style.RESET_ALL}"
    else:
        return f"{DIM_COLOR}・ 陌生{Style.RESET_ALL}"


# ============================================================
# 消息流式输出
# ============================================================

def stream_chat(client: OpenAI, state: ConversationState, user_input: str):
    """发送消息并流式输出三玖的回复"""
    # 将用户消息加入历史
    state.messages.append({"role": "user", "content": user_input})

    # 构建请求消息列表
    # 基础 system prompt（含好感度调整）
    base_prompt = get_system_prompt(state.affection)
    # 叠加世界书匹配的背景知识
    system_prompt = build_enhanced_prompt(base_prompt, user_input)
    api_messages = [{"role": "system", "content": system_prompt}]

    # 只取最近 N 轮历史，避免超出 token 限制
    recent = state.messages[-(MAX_HISTORY_TURNS * 2):]
    api_messages.extend(recent)

    # 调用 API
    params = get_chat_params()
    params.pop("stream", None)  # stream 单独传

    try:
        stream = client.chat.completions.create(
            **params,
            messages=api_messages,
            stream=True,
        )

        # 流式输出
        print(f"\n{MIKU_COLOR}三玖: {Style.RESET_ALL}", end="", flush=True)
        full_response = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(f"{MIKU_COLOR}{content}{Style.RESET_ALL}", end="", flush=True)
                full_response += content

        print()  # 换行

        # 将三玖的回复加入历史
        if full_response.strip():
            state.messages.append({"role": "assistant", "content": full_response.strip()})
        else:
            state.messages.append(
                {"role": "assistant", "content": "…（脸红低头）…///…"}
            )

        return full_response

    except Exception as e:
        error_msg = str(e)
        print(f"\n{ERROR_COLOR}❌ 连接失败: {error_msg}")

        if "401" in error_msg or "Unauthorized" in error_msg:
            print(f"{ERROR_COLOR}   → API Key 无效，请检查 .env 文件中的 DEEPSEEK_API_KEY")
        elif "402" in error_msg or "Insufficient" in error_msg:
            print(f"{ERROR_COLOR}   → 账户余额不足，请前往 platform.deepseek.com 充值")
        elif "429" in error_msg:
            print(f"{ERROR_COLOR}   → 请求太频繁，稍等一下再试")
        elif "Connection" in error_msg or "timeout" in error_msg:
            print(f"{ERROR_COLOR}   → 网络连接失败，请检查网络或代理设置")

        return None


# ============================================================
# 本地模型对话
# ============================================================

def stream_local_chat(model, state: ConversationState, user_input: str):
    """使用本地 QLoRA 模型生成三玖回复（带 TextStreamer 流式输出）"""
    # 将用户消息加入历史
    state.messages.append({"role": "user", "content": user_input})

    # 构建消息列表
    # 本地模型用精简 system prompt（全文 ~3000 字会溢出 512 token 窗口）
    base_prompt = get_system_prompt(state.affection)
    short_prompt = (
        "你是中野三玖，17岁，内向害羞。喜欢战国历史和抹茶。说话温柔、带日语语气词。"
        + base_prompt[-200:]  # 取当前状态提示（好感度相关，约200字）
    )
    system_prompt = build_enhanced_prompt(short_prompt, user_input)

    # 如果检测到用户情绪低落，追加安慰提示
    if state.user_emotion == "negative":
        system_prompt += "\n\n对方现在情绪不太好，请温柔地安慰和陪伴，不要讲大道理。"

    api_messages = [{"role": "system", "content": system_prompt}]
    recent = state.messages[-(MAX_HISTORY_TURNS * 2):]
    api_messages.extend(recent)

    # 调用本地模型
    print(f"\n{MIKU_COLOR}三玖: {Style.RESET_ALL}", end="", flush=True)
    reply = model.chat(
        api_messages,
        stream=True,
        max_new_tokens=LOCAL_MAX_NEW_TOKENS,
        temperature=LOCAL_TEMPERATURE,
        top_p=LOCAL_TOP_P,
    )

    # 将回复加入历史
    if reply.strip():
        state.messages.append({"role": "assistant", "content": reply.strip()})
    else:
        state.messages.append(
            {"role": "assistant", "content": "…（脸红低头）…///…"}
        )

    return reply


# ============================================================
# 命令处理
# ============================================================

def handle_command(cmd: str, state: ConversationState, client: OpenAI | None = None) -> bool:
    """
    处理聊天命令
    返回 True 表示继续聊天，False 表示退出
    client 参数仅在 API 模式下传入，本地模式下为 None
    """
    cmd = cmd.strip().lower()

    if cmd in ("/exit", "/bye", "/quit", "/q"):
        # 三玖道别
        affection = state.affection
        if affection >= 70:
            farewell = "…要走了吗…？那…明天也要来找我哦…我会等你的…///…おやすみ…"
        elif affection >= 30:
            farewell = "嗯…今天聊得很开心…那个…下次再一起聊天吧…またね…"
        else:
            farewell = "…再见…今天谢谢你陪我说话…"
        print(f"\n{MIKU_COLOR}三玖: {farewell}{Style.RESET_ALL}\n")
        save_conversation(state)
        print(f"{DIM_COLOR}[保存] 对话已保存。下次继续哦～{Style.RESET_ALL}")
        return False

    elif cmd == "/save":
        save_conversation(state)
        print(f"{SYSTEM_COLOR}[保存] 对话已手动保存{Style.RESET_ALL}")
        return True

    elif cmd == "/clear":
        state.messages = []
        state.total_turns = 0
        state.affection = 0
        state.mood = "shy"
        save_conversation(state)
        print(f"{SYSTEM_COLOR}[清除] 对话历史已清除，三玖忘记了之前的对话…（好感度重置）{Style.RESET_ALL}")
        # 模拟三玖的反应
        print(f"{MIKU_COLOR}三玖: …诶？我们…刚才在聊什么来着…///…{Style.RESET_ALL}\n")
        return True

    elif cmd == "/status":
        emoji = get_mood_emoji(state.mood)
        label = get_affection_label(state.affection)
        user_emoji = get_mood_emoji(state.user_emotion if state.user_emotion != "neutral" else "neutral")
        print(f"\n{SYSTEM_COLOR}┌───────────── 三玖的状态 ─────────────┐")
        print(f"│  好感度: {state.affection}/100  {label}")
        print(f"│  心  情: {emoji}  {state.mood}")
        print(f"│  对方情绪: {user_emoji}  {state.user_emotion}")
        print(f"│  对话轮: {state.total_turns} 轮")
        print(f"│  记  忆: {len(state.messages)} 条消息 ({len(state.messages)//2} 轮可用)")
        print(f"└──────────────────────────────────────┘{Style.RESET_ALL}\n")
        return True

    elif cmd == "/help" or cmd == "/?":
        print(f"""
{SYSTEM_COLOR}┌───────────── 可用命令 ─────────────┐
│  /save      手动保存对话            │
│  /clear     清除记忆（重置好感度）     │
│  /status    查看三玖的状态           │
│  /exit      退出（三玖会跟你道别）     │
│  /bye       同上                   │
│  /help      显示此帮助              │
└────────────────────────────────────┘{Style.RESET_ALL}
""")
        return True

    elif cmd.startswith("/"):
        print(f"{ERROR_COLOR}未知命令: {cmd}。输入 /help 查看可用命令{Style.RESET_ALL}")
        return True

    # 不是命令，是普通消息
    return None


# ============================================================
# 启动画面
# ============================================================

def show_welcome(state: ConversationState):
    """显示启动画面"""
    print()
    print(f"{MIKU_COLOR}")
    print("    +==================================+")
    print("    |       中野 三玖                  |")
    print("    |   Nakano Miku  |  なかの みく   |")
    print("    +==================================+")
    print(f"{Style.RESET_ALL}")

    if state.total_turns > 0:
        emoji = get_mood_emoji(state.mood)
        label = get_affection_label(state.affection)
        print(f"{DIM_COLOR}  欢迎回来～ 好感度: {state.affection}/100 {label} {emoji}")
        print(f"  你们已经聊过 {state.total_turns} 轮了{Style.RESET_ALL}")
        print()
        # 三玖的回归问候
        if state.affection >= 50:
            print(f"{MIKU_COLOR}三玖: 啊…你来了…那个…一直在等你…///…今天也想和你聊天…{Style.RESET_ALL}")
        else:
            print(f"{MIKU_COLOR}三玖: あの…又见面了呢…请多关照…{Style.RESET_ALL}")
    else:
        print(f"{DIM_COLOR}  输入消息开始和三玖聊天吧～")
        print(f"  输入 /help 查看可用命令{Style.RESET_ALL}")
        print()
        print(f"{MIKU_COLOR}三玖: …（抬头看了你一眼，又低头摆弄耳机）…こんにちは…我是三玖…请多指教…{Style.RESET_ALL}")

    print()


# ============================================================
# 主循环
# ============================================================

def main():
    """主程序入口 — 支持本地模型 / DeepSeek API 双模式"""
    client = None
    model = None

    if USE_LOCAL_MODEL:
        # === 本地模型模式（懒加载）===
        from local_model import get_model
        print(f"{SYSTEM_COLOR}[加载] 正在加载本地模型...{Style.RESET_ALL}")
        try:
            model = get_model()
        except Exception as e:
            print(f"{ERROR_COLOR}❌ 本地模型加载失败: {e}{Style.RESET_ALL}")
            print(f"{ERROR_COLOR}   请检查模型文件是否完整{Style.RESET_ALL}\n")
            return
        print(f"{SYSTEM_COLOR}[就绪] 本地模型就绪{Style.RESET_ALL}")
    else:
        # === API 模式 ===
        try:
            client = create_client()
        except ValueError as e:
            print(f"\n{ERROR_COLOR}{e}{Style.RESET_ALL}\n")
            return

    # 加载历史对话
    state = load_conversation()
    if not state.created_at:
        state.created_at = datetime.now().isoformat()

    # 显示启动画面
    show_welcome(state)

    # 主聊天循环
    while True:
        try:
            user_input = input(f"{USER_COLOR}你: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM_COLOR}收到退出信号…{Style.RESET_ALL}")
            user_input = "/exit"

        if not user_input:
            continue

        # 命令处理
        cmd_result = handle_command(user_input, state, client)
        if cmd_result is False:
            break
        elif cmd_result is True:
            continue

        # 更新好感度（双轨：关键词 + SnowNLP）
        update_affection(state, user_input)

        # 发送消息并获取回复
        if USE_LOCAL_MODEL:
            stream_local_chat(model, state, user_input)
        else:
            stream_chat(client, state, user_input)

        # 更新轮次
        state.total_turns += 1

        # 自动保存
        if AUTO_SAVE and state.total_turns % 5 == 0:
            save_conversation(state)

    # 退出
    print(f"{DIM_COLOR}またね～ (mata ne~){Style.RESET_ALL}")


if __name__ == "__main__":
    main()
