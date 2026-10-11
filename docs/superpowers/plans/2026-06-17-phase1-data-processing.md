# Phase 1 数据处理 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建训练数据集：清洗 LCCC-base 中文对话 + 生成 1000 条三玖角色对话 + 合并划分 train/val

**Architecture:** 两个独立脚本并行开发。`preprocess.py` 负责 LCCC 下载+清洗+合并划分；`miku_data_generator.py` 负责调用 DeepSeek API 生成三玖风格数据。中间产物落盘，最终产出 `data/train.jsonl` 和 `data/val.jsonl`。

**Tech Stack:** Python, datasets (HuggingFace), jieba, OpenAI SDK (DeepSeek), jsonl

**前置条件:** Phase 0 完成，依赖已安装。KMP_DUPLICATE_LIB_OK=TRUE 环境变量需设置。

**依赖关系:** Task 1 → {Task 2, Task 3}（并行）→ Task 4 → Task 5

---

### Task 1: 目录结构 + 停用词表下载

**Files:**
- Create: `data/stopwords/hit_stopwords.txt`
- Create: `data/stopwords/__init__.py` (空文件，标记为 package)

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p "D:\Pythonpro\artificial\Al-chat\data\stopwords"
mkdir -p "D:\Pythonpro\artificial\Al-chat\data\raw"
```

- [ ] **Step 2: 下载哈工大停用词表**

```bash
curl -L -o "D:\Pythonpro\artificial\Al-chat\data\stopwords\hit_stopwords.txt" \
  "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt"
```

- [ ] **Step 3: 验证停用词表**

```bash
wc -l "D:\Pythonpro\artificial\Al-chat\data\stopwords\hit_stopwords.txt"
```
Expected: ~1000+ 行

- [ ] **Step 4: 验证目录结构**

```bash
ls -la "D:\Pythonpro\artificial\Al-chat\data/"
ls -la "D:\Pythonpro\artificial\Al-chat\data\stopwords/"
```
Expected: `data/raw/` 和 `data/stopwords/hit_stopwords.txt` 存在

---

### Task 2: 三玖风格数据生成器

**Files:**
- Create: `data/miku_data_generator.py`
- Output: `data/miku_style_data.jsonl`

- [ ] **Step 1: 创建 miku_data_generator.py — 导入与配置加载**

写入以下完整脚本：

```python
"""
三玖风格对话数据生成器
使用 DeepSeek API 批量生成角色对话，带断点续跑和重试机制。
输出: data/miku_style_data.jsonl (1000条)
"""
import json
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# === 路径设置 ===
PROJECT_ROOT = Path(__file__).parent.parent
AIMIKU_ROOT = Path(r"D:\AImiku")
OUTPUT_FILE = PROJECT_ROOT / "data" / "miku_style_data.jsonl"

# === 加载 API Key ===
load_dotenv(AIMIKU_ROOT / ".env")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

if not API_KEY:
    raise ValueError("未找到 DEEPSEEK_API_KEY，请检查 D:\\AImiku\\.env")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === 导入三玖系统提示（从本地复制版本）===
sys.path.insert(0, str(PROJECT_ROOT))
from miku_character import MIKU_SYSTEM_PROMPT

# === 50个种子场景 ===
SEED_SCENARIOS = [
    # 日常问候和闲聊 (10个)
    "你好呀，今天过得怎么样？",
    "晚上好，你在做什么呢？",
    "早上好！今天天气不错呢",
    "嗨，好久不见，最近还好吗？",
    "今天上班好累啊…想找人聊聊天",
    "周末有什么计划吗？",
    "吃饭了吗？我在想要不要做点抹茶点心",
    "今天下雨了，适合待在家里看书",
    "最近有没有看什么有意思的电视剧？",
    "放学了吗？今天的课难不难？",

    # 分享心情 (10个)
    "我今天超级开心！发生了一件好事",
    "最近压力好大，有点喘不过气",
    "有点想家了…虽然才离开没多久",
    "今天被老板夸了，好高兴！",
    "唉，今天心情不太好，不想说话",
    "突然觉得有点孤单",
    "今天和朋友吵架了，好难过",
    "最近一切都很顺利，心情特别好",
    "不知道为什么，今天就是觉得很幸福",
    "感觉最近做什么都不顺心",

    # 战国历史/抹茶话题 (10个)
    "你对日本战国时代了解吗？我想听听",
    "织田信长和丰臣秀吉，你觉得谁更厉害？",
    "今天买到了很好的抹茶粉！要不要一起喝？",
    "你最喜欢的战国武将是谁？为什么？",
    "抹茶冰淇淋和抹茶蛋糕，你更喜欢哪个？",
    "本能寺之变真的好戏剧性啊，你怎么看？",
    "教我做抹茶点心好不好？",
    "你觉得武田信玄和上杉谦信谁更强？",
    "我最近在看关于战国时代的小说，好有趣",
    "什么样的抹茶才算好抹茶？教教我怎么分辨",

    # 寻求安慰 (10个)
    "今天被老师批评了，好难过…",
    "我觉得自己什么都做不好",
    "和最好的朋友冷战了，不知道怎么办",
    "考试没考好，努力了这么久却…",
    "家里人吵架了，我夹在中间好难受",
    "感觉自己不被需要",
    "工作上遇到了很大的挫折",
    "失恋了，好痛苦",
    "被别人误会了，解释也没用",
    "今天好倒霉，什么都不顺",

    # 夸奖/吐槽三玖 (10个)
    "三玖你今天穿得很好看！",
    "三玖你说话好温柔啊",
    "三玖你怎么老是戴着耳机？不闷吗？",
    "三玖你知道你害羞的样子很可爱吗？",
    "三玖你讲历史的时候整个人都不一样了呢",
    "三玖你是不是又躲在角落里看历史书了？",
    "三玖你做的料理真的进步了好多！",
    "三玖你怎么又脸红了哈哈",
    "三玖你其实比你自己想象的要厉害多了",
    "三玖你是不是偷偷喜欢看恋爱漫画？",
]


def generate_one(scenario: str, retries: int = 3) -> str | None:
    """调用 DeepSeek API，用三玖身份回复一个场景。返回回复文本或 None。"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": MIKU_SYSTEM_PROMPT},
                    {"role": "user", "content": f"（这是对话练习，请用三玖的语气自然地回复下面这句话）\n{scenario}"},
                ],
                temperature=0.9,
                max_tokens=256,
                stream=False,
            )
            reply = response.choices[0].message.content.strip()
            if reply:
                return reply
        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠ 重试 {attempt+1}/{retries}，等待 {wait}s: {e}")
            time.sleep(wait)
    return None


def load_checkpoint() -> list[dict]:
    """加载已有数据（断点续跑）"""
    if OUTPUT_FILE.exists():
        items = []
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    return []


def save_items(items: list[dict]):
    """全量保存到 jsonl"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    total = len(SEED_SCENARIOS) * 20  # 每场景20条 = 1000条
    existing = load_checkpoint()
    existing_count = len(existing)

    if existing_count >= total:
        print(f"✅ 已有 {existing_count} 条数据，跳过生成。")
        return

    print(f"📋 目标 {total} 条，已有 {existing_count} 条，还需 {total - existing_count} 条")
    print(f"🔑 使用模型: {MODEL}")

    items = existing[:]
    scenario_index = existing_count // 20  # 从哪个场景开始
    turn_index = existing_count % 20       # 该场景的第几条

    for si in range(scenario_index, len(SEED_SCENARIOS)):
        scenario = SEED_SCENARIOS[si]
        start_turn = turn_index if si == scenario_index else 0
        for ti in range(start_turn, 20):
            current = len(items)
            print(f"[{current+1}/{total}] 场景 {si+1}/50 第{ti+1}/20条: {scenario[:30]}...", end=" ", flush=True)

            reply = generate_one(scenario)
            if reply:
                items.append({
                    "messages": [
                        {"role": "user", "content": scenario},
                        {"role": "assistant", "content": reply},
                    ]
                })
                print(f"✅ ({len(reply)}字)")
            else:
                print("❌ 跳过")

            # 每10条保存一次
            if (current + 1) % 10 == 0:
                save_items(items)
                print(f"  💾 已保存 {len(items)} 条")

            time.sleep(0.5)  # 速率限制

    # 最终保存
    save_items(items)
    print(f"\n🎉 完成！共生成 {len(items)} 条三玖风格对话")
    print(f"📁 保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本语法**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "import py_compile; py_compile.compile('data/miku_data_generator.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: 运行生成器（尝鲜版 — 只生成 2 条验证流程）**

临时在脚本末尾 `if __name__ == "__main__":` 之前插入测试逻辑，或者直接测试 API 连通性：

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path; from dotenv import load_dotenv; import os
load_dotenv(Path(r'D:\AImiku') / '.env')
from openai import OpenAI
from miku_character import MIKU_SYSTEM_PROMPT
client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'))
r = client.chat.completions.create(model=os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-pro'), messages=[
    {'role': 'system', 'content': MIKU_SYSTEM_PROMPT},
    {'role': 'user', 'content': '你好呀'}
], temperature=0.9, max_tokens=128)
print('API OK:', r.choices[0].message.content[:80])
"
```
Expected: 输出三玖风格的中文回复

- [ ] **Step 4: 运行完整生成（后台执行）**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python data/miku_data_generator.py
```

> ⚠ 此步骤预计耗时 5-15 分钟（1000 次 API 调用）。断点续跑：中断后重新运行即可从上次位置继续。

- [ ] **Step 5: 验证输出**

```bash
wc -l "D:\Pythonpro\artificial\Al-chat\data\miku_style_data.jsonl"
```
Expected: ≥ 950 行

```bash
python -c "
import json
with open('D:\Pythonpro\artificial\Al-chat\data\miku_style_data.jsonl', 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f if line.strip()]
print(f'共 {len(items)} 条')
# 验证格式
for item in items[:3]:
    msgs = item['messages']
    assert len(msgs) == 2
    assert msgs[0]['role'] == 'user'
    assert msgs[1]['role'] == 'assistant'
    print(f'  User: {msgs[0][\"content\"][:40]}...')
    print(f'  Miku: {msgs[1][\"content\"][:40]}...')
    print()
print('格式验证通过 ✅')
"
```

---

### Task 3: LCCC 数据下载与清洗

**Files:**
- Create: `data/preprocess.py`
- Output: `data/clean_data.jsonl`

- [ ] **Step 1: 创建 preprocess.py — 完整脚本**

```python
"""
LCCC-base 中文对话数据清洗流水线
下载 → 正则清洗 → jieba分词→ 去停用词过滤 → 长度过滤 → 格式化 → 保存
"""
import json
import re
import os
import time
from pathlib import Path
from collections import Counter

# === 路径设置 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
STOPWORDS_FILE = DATA_DIR / "stopwords" / "hit_stopwords.txt"
CLEAN_OUTPUT = DATA_DIR / "clean_data.jsonl"
MIKU_DATA = DATA_DIR / "miku_style_data.jsonl"
TRAIN_OUTPUT = DATA_DIR / "train.jsonl"
VAL_OUTPUT = DATA_DIR / "val.jsonl"


def download_lccc():
    """下载 LCCC-base 数据集，返回对话列表"""
    print("=" * 60)
    print("📥 Step 1: 下载 LCCC-base 数据集")
    from datasets import load_dataset
    dataset = load_dataset("lccc", "base", cache_dir=str(RAW_DIR))
    # LCCC-base 结构: {'train': [...], 'test': [...]} 或只有 train
    # 每项可能是 dict，格式因版本而异；一般有 "messages" 或 "dialog"/"content"
    conversations = []
    for split in dataset.keys():
        for item in dataset[split]:
            # LCCC-base 返回的格式: {"dialog": [{"role":"user","content":"..."}, ...]} 
            # 或直接是 list of turns
            if isinstance(item, dict):
                if "messages" in item:
                    conversations.append(item["messages"])
                elif "dialog" in item:
                    conversations.append(item["dialog"])
                elif "content" in item:
                    # 某些版本直接是单轮文本
                    conversations.append(item["content"])
            elif isinstance(item, list):
                conversations.append(item)
    print(f"   原始对话总数: {len(conversations)}")
    return conversations


def load_stopwords() -> set:
    """加载停用词表"""
    stopwords = set()
    if STOPWORDS_FILE.exists():
        with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        print(f"   加载停用词: {len(stopwords)} 个")
    else:
        # 内置基础停用词兜底
        stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
            "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "什么",
            "怎么", "哪", "吗", "啊", "吧", "呢", "哦", "嗯", "呀", "哈", "哇",
        }
        print(f"   ⚠ 停用词表未找到，使用内置 {len(stopwords)} 个基础词兜底")
    return stopwords


def regex_clean(text: str) -> str:
    """正则清洗单条文本"""
    # 去 URL
    text = re.sub(r'https?://\S+', '', text)
    # 去 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去特殊 Unicode 控制字符（保留中文、字母、数字、常用标点）
    text = re.sub(r'[^一-鿿　-〿＀-￯a-zA-Z0-9，。！？、；：""''（）…—\s]', '', text)
    return text.strip()


def is_pure_punctuation(text: str) -> bool:
    """检查是否纯标点/空白"""
    cleaned = re.sub(r'[\s，。！？、；：""''（）…—　-〿＀-￯]+', '', text)
    return len(cleaned) == 0


def stopword_ratio(text: str, stopwords: set) -> float:
    """计算停用词占比（先 jieba 分词）"""
    import jieba
    tokens = list(jieba.cut(text))
    if len(tokens) == 0:
        return 1.0
    stop_count = sum(1 for t in tokens if t.strip() in stopwords)
    return stop_count / len(tokens)


def extract_turns(conv) -> list[dict] | None:
    """
    从 LCCC 的各种格式中提取 user/assistant 对话对。
    返回 [{"role":"user","content":"..."},{"role":"assistant","content":"..."}] 列表
    """
    if isinstance(conv, list) and len(conv) >= 2:
        turns = []
        for item in conv:
            if isinstance(item, dict) and "role" in item and "content" in item:
                turns.append(item)
        return turns if len(turns) >= 2 else None
    return None


def clean_conversations(raw_convs: list, stopwords: set) -> list[dict]:
    """
    清洗流水线：
    1. 正则清洗每句话
    2. 去除纯标点行
    3. 停用词占比过滤（>50% 则丢弃整条对话）
    4. 长度过滤（4-50字）
    5. 格式化为 OpenAI messages 格式
    """
    print("\n" + "=" * 60)
    print("🧹 Step 2: 数据清洗")

    stats = Counter()
    cleaned = []

    for i, conv in enumerate(raw_convs):
        stats["total"] += 1

        turns = extract_turns(conv)
        if turns is None:
            stats["bad_format"] += 1
            continue

        # 为每轮对话提取 user-assistant 对
        all_msgs = []
        for turn in turns:
            content = regex_clean(turn.get("content", ""))
            role = turn.get("role", "")

            # 纯标点过滤
            if is_pure_punctuation(content) or len(content) == 0:
                continue

            # 长度过滤
            if len(content) < 4 or len(content) > 50:
                continue

            all_msgs.append({"role": role, "content": content})

        if len(all_msgs) < 2:
            stats["too_short"] += 1
            continue

        # 停用词占比过滤：计算整个对话中 user/assistant 的平均停用词占比
        ratios = [stopword_ratio(m["content"], stopwords) for m in all_msgs if m["role"] in ("user", "assistant")]
        if not ratios:
            stats["no_valid"] += 1
            continue
        avg_ratio = sum(ratios) / len(ratios)
        if avg_ratio > 0.5:
            stats["high_stopword"] += 1
            continue

        # 只保留 user-assistant 对的相邻对话（取前两条）
        user_msgs = [m for m in all_msgs if m["role"] == "user"]
        asst_msgs = [m for m in all_msgs if m["role"] == "assistant"]
        if user_msgs and asst_msgs:
            cleaned.append({
                "messages": [
                    {"role": "user", "content": user_msgs[0]["content"]},
                    {"role": "assistant", "content": asst_msgs[0]["content"]},
                ]
            })
            stats["kept"] += 1

        # 进度
        if (i + 1) % 10000 == 0:
            print(f"   已处理 {i+1}/{len(raw_convs)}... 保留 {stats['kept']}")

    print(f"\n   原始: {stats['total']}")
    print(f"   格式错误: {stats['bad_format']}")
    print(f"   过短: {stats['too_short']}")
    print(f"   无效内容: {stats['no_valid']}")
    print(f"   停用词过多: {stats['high_stopword']}")
    print(f"   ✅ 保留: {stats['kept']}")
    return cleaned


def save_clean_data(items: list[dict]):
    """保存清洗后的数据"""
    with open(CLEAN_OUTPUT, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n💾 清洗数据已保存: {CLEAN_OUTPUT} ({len(items)} 条)")


def merge_and_split():
    """合并 LCCC 清洗数据 + 三玖风格数据，随机划分 train/val (9:1)"""
    import random
    random.seed(42)

    print("\n" + "=" * 60)
    print("🔀 Step 3: 合并与划分")

    # 加载通用对话
    lccc_items = []
    if CLEAN_OUTPUT.exists():
        with open(CLEAN_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lccc_items.append(json.loads(line))
    print(f"   LCCC 清洗数据: {len(lccc_items)} 条")

    # 加载三玖风格数据
    miku_items = []
    if MIKU_DATA.exists():
        with open(MIKU_DATA, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    miku_items.append(json.loads(line))
    print(f"   三玖风格数据: {len(miku_items)} 条")

    # 合并
    all_items = lccc_items + miku_items
    random.shuffle(all_items)

    # 9:1 划分
    split_idx = int(len(all_items) * 0.9)
    train = all_items[:split_idx]
    val = all_items[split_idx:]

    # 保存
    def save_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(TRAIN_OUTPUT, train)
    save_jsonl(VAL_OUTPUT, val)

    print(f"   📊 训练集: {len(train)} 条 → {TRAIN_OUTPUT}")
    print(f"   📊 验证集: {len(val)} 条 → {VAL_OUTPUT}")
    print(f"   📐 比例: {len(train)/len(all_items)*100:.1f}% / {len(val)/len(all_items)*100:.1f}%")


def print_samples(filepath: Path, n: int = 3, label: str = ""):
    """打印样本"""
    print(f"\n{'='*60}")
    print(f"📋 样本检查: {label}")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        for i, line in enumerate(lines[:n]):
            item = json.loads(line)
            msgs = item["messages"]
            print(f"\n  --- 样本 {i+1} ---")
            for m in msgs:
                print(f"  [{m['role']}]: {m['content'][:60]}...")


def main():
    import jieba
    start = time.time()

    # Step 1: 下载
    raw_convs = download_lccc()

    # Step 2: 清洗
    stopwords = load_stopwords()
    cleaned = clean_conversations(raw_convs, stopwords)
    save_clean_data(cleaned)

    # 打印清洗样本
    if cleaned:
        print_samples(CLEAN_OUTPUT, n=3, label="LCCC 清洗后")

    # Step 3: 合并划分
    merge_and_split()

    # 打印最终样本
    print_samples(TRAIN_OUTPUT, n=3, label="训练集")
    print_samples(VAL_OUTPUT, n=3, label="验证集")

    elapsed = time.time() - start
    print(f"\n⏱ 总耗时: {elapsed:.1f}s")
    print("🎉 Phase 1 数据处理完成！")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本语法**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "import py_compile; py_compile.compile('data/preprocess.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: 运行清洗流水线**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python data/preprocess.py
```
Expected: 逐步输出日志：下载进度 → 清洗统计 → 合并划分 → 完成

> ⚠ LCCC 数据集首次下载可能需要 2-5 分钟（约 100-300MB）。之后使用缓存。

- [ ] **Step 4: 验证输出文件**

```bash
wc -l "D:\Pythonpro\artificial\Al-chat\data\clean_data.jsonl"
wc -l "D:\Pythonpro\artificial\Al-chat\data\train.jsonl"
wc -l "D:\Pythonpro\artificial\Al-chat\data\val.jsonl"
```
Expected:
- `clean_data.jsonl` ≥ 50000 行
- `train.jsonl` ≈ `clean_data` + 三玖数据的 90%
- `val.jsonl` ≈ 10%

---

### Task 4: 集成验证

**Files:** 无新建，验证已有产出

- [ ] **Step 1: 综合验证脚本**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "
import json
from pathlib import Path

data_dir = Path('data')
checks = {}

# 1. 文件存在性
for fn in ['clean_data.jsonl', 'miku_style_data.jsonl', 'train.jsonl', 'val.jsonl']:
    p = data_dir / fn
    exists = p.exists()
    count = sum(1 for _ in open(p, 'r', encoding='utf-8')) if exists else 0
    checks[fn] = {'exists': exists, 'lines': count}
    print(f'📁 {fn}: {\"✅\" if exists else \"❌\"} {count} 行')

# 2. 验证标准检查
assert checks['clean_data.jsonl']['lines'] >= 50000, 'clean_data 不足 50000 行'
assert checks['miku_style_data.jsonl']['lines'] >= 950, 'miku_style_data 不足 950 行'

train = checks['train.jsonl']['lines']
val = checks['val.jsonl']['lines']
ratio = val / (train + val) if (train + val) > 0 else 0
print(f'📐 训练/验证比例: {train}/{val} = {ratio:.2%} (期望 ~10% val)')
assert 0.08 <= ratio <= 0.12, f'验证集比例异常: {ratio:.2%}'

# 3. 格式验证
for fn in ['train.jsonl', 'val.jsonl']:
    with open(data_dir / fn, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            item = json.loads(line)
            msgs = item['messages']
            assert len(msgs) >= 2, f'{fn}:{i} 对话轮数不足'
            for m in msgs:
                assert 'role' in m and 'content' in m, f'{fn}:{i} 格式错误'
                assert m['role'] in ('user', 'assistant', 'system'), f'{fn}:{i} 未知角色'
    print(f'📋 {fn} 格式验证通过 (抽检100条)')

# 4. 打印训练集样例
print('\n--- 训练集样例 ---')
with open(data_dir / 'train.jsonl', 'r', encoding='utf-8') as f:
    item = json.loads(f.readline())
    for m in item['messages']:
        print(f'  [{m[\"role\"]}]: {m[\"content\"][:60]}...')

print('\n🎉 全部验证通过！')
"
```
Expected: 所有检查项 ✅，无 AssertionError

- [ ] **Step 2: 人工抽检 5 条三玖数据**

```bash
cd "D:\Pythonpro\artificial\Al-chat" && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "
import json, random
random.seed(123)
with open('data/miku_style_data.jsonl', 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f if line.strip()]
samples = random.sample(items, min(5, len(items)))
for i, item in enumerate(samples):
    print(f'\\n=== 样本 {i+1} ===')
    for m in item['messages']:
        print(f'[{m[\"role\"]}]: {m[\"content\"]}')
"
```
Expected: 回复内容与三玖角色设定一致（含有日语语气词、害羞风格、不超过 256 tokens）

---

### Task 5: 最终检查清单

- [ ] `data/clean_data.jsonl` ≥ 50000 行
- [ ] `data/miku_style_data.jsonl` ≥ 950 行
- [ ] `data/train.jsonl` 和 `data/val.jsonl` 比例接近 9:1
- [ ] 所有 jsonl 格式正确（每行一个 JSON，含 `messages` 字段）
- [ ] 三玖回复风格符合角色设定（人工抽检 5 条）
- [ ] jieba 分词和去停用词日志在 `preprocess.py` 输出中可见
- [ ] 清洗统计合理（保留率在合理范围，非 0 非 100%）
