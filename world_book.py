"""
三玖世界书引擎
───────────────
加载 world_book.json，根据对话内容匹配并激活相关世界观条目。
将匹配到的背景信息注入上下文，帮助 AI 保持角色一致性。

支持:
- 关键词匹配（中文 + 日语 + 英文）
- 优先级排序
- 常驻条目（constant = true 的始终激活）
- 最大条目限制（避免 context 过长）
"""

import json
import re
from pathlib import Path
from typing import Optional


# === 世界书数据 ===
WORLD_BOOK_FILE = Path(__file__).parent / "world_book.json"

# 单次最多注入的触发条目数
MAX_ACTIVE_ENTRIES = 5

# 常驻条目数上限
MAX_CONSTANT_ENTRIES = 3


class WorldBookEntry:
    """世界书单条目"""

    def __init__(self, data: dict):
        self.id: str = data.get("id", "")
        self.category: str = data.get("category", "general")
        self.keys: list[str] = [k.lower() for k in data.get("keys", [])]
        self.content: str = data.get("content", "")
        self.priority: int = data.get("priority", 0)
        self.constant: bool = data.get("constant", False)

    def matches(self, text: str) -> bool:
        """检查文本是否触发此条目（任一关键词匹配）"""
        text_lower = text.lower()
        for key in self.keys:
            if key in text_lower:
                return True
        return False

    def __repr__(self):
        const_mark = "📍" if self.constant else "  "
        return f"[{const_mark} p{self.priority:03d}] {self.id}"


class WorldBook:
    """世界书管理器"""

    def __init__(self, file_path: Path = WORLD_BOOK_FILE):
        self.entries: list[WorldBookEntry] = []
        self.constant_entries: list[WorldBookEntry] = []
        self.dynamic_entries: list[WorldBookEntry] = []
        self.loaded = False
        self._load(file_path)

    def _load(self, file_path: Path):
        """加载世界书 JSON 文件"""
        if not file_path.exists():
            print(f"[WorldBook] WARN File not found: {file_path}")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WorldBook] ERROR Load failed: {e}")
            return

        entries_data = data.get("entries", [])
        for entry_data in entries_data:
            entry = WorldBookEntry(entry_data)
            self.entries.append(entry)
            if entry.constant:
                self.constant_entries.append(entry)
            else:
                self.dynamic_entries.append(entry)

        # 排序：优先级高的在前
        self.constant_entries.sort(key=lambda e: e.priority, reverse=True)
        self.dynamic_entries.sort(key=lambda e: e.priority, reverse=True)
        self.entries.sort(key=lambda e: e.priority, reverse=True)

        self.loaded = True
        meta = data.get("meta", {})
        print(f"[WorldBook] [OK] Loaded '{meta.get('name', 'Unknown')}': "
              f"{len(self.entries)} entries "
              f"({len(self.constant_entries)} constant, {len(self.dynamic_entries)} dynamic)")

    def query(self, text: str, max_entries: int = MAX_ACTIVE_ENTRIES) -> list[WorldBookEntry]:
        """
        根据输入文本查询匹配的世界书条目

        返回: 匹配的动态条目列表（按优先级排序）
        """
        if not text.strip():
            return []

        matched = []
        for entry in self.dynamic_entries:
            if entry.matches(text):
                matched.append(entry)

        # 按优先级排序，限制数量
        matched.sort(key=lambda e: e.priority, reverse=True)
        return matched[:max_entries]

    def get_context_block(self, matched_entries: list[WorldBookEntry]) -> str:
        """
        将匹配的条目拼接为上下文文本块

        写入 system prompt 前，作为三玖需要知道的「世界观背景知识」
        """
        if not matched_entries:
            return ""

        lines = ["\n【世界观背景 - 以下信息你已知晓，融入角色自然使用】"]
        for entry in matched_entries:
            lines.append(f"- {entry.content}")

        return "\n".join(lines)

    def get_constant_block(self) -> str:
        """获取常驻条目的上下文文本块（始终注入）"""
        if not self.constant_entries:
            return ""

        top = self.constant_entries[:MAX_CONSTANT_ENTRIES]
        lines = ["\n【核心设定 - 始终遵守】"]
        for entry in top:
            lines.append(f"- {entry.content}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """返回统计信息"""
        return {
            "total": len(self.entries),
            "constant": len(self.constant_entries),
            "dynamic": len(self.dynamic_entries),
            "categories": self._count_categories(),
        }

    def _count_categories(self) -> dict:
        counts = {}
        for e in self.entries:
            counts[e.category] = counts.get(e.category, 0) + 1
        return counts


# ============================================================
# 全局单例
# ============================================================

_world_book: Optional[WorldBook] = None


def get_world_book() -> WorldBook:
    """获取世界书单例"""
    global _world_book
    if _world_book is None:
        _world_book = WorldBook()
    return _world_book


# ============================================================
# 便捷函数
# ============================================================

def build_enhanced_prompt(base_system_prompt: str, user_input: str) -> str:
    """
    基于用户输入，构建增强版 system prompt

    将世界书中匹配到的背景知识注入到基础 prompt 后面。
    聊天程序直接调用此函数即可。
    """
    wb = get_world_book()

    if not wb.loaded or not wb.entries:
        return base_system_prompt

    parts = [base_system_prompt]

    # 1. 常驻条目
    constant_block = wb.get_constant_block()
    if constant_block:
        parts.append(constant_block)

    # 2. 动态匹配条目
    matched = wb.query(user_input)
    if matched:
        context = wb.get_context_block(matched)
        parts.append(context)

    return "\n".join(parts)
