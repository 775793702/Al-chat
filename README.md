# 三玖 AI —— 基于 QLoRA 微调的中文角色对话模型

> 用消费级显卡（8 GB）微调 15 亿参数大模型，打造具有角色人格的中文对话机器人。

本项目完整走通了「数据清洗 → QLoRA 微调 → 系统集成 → 评估」的全流程，以动漫《五等分的新娘》角色**中野三玖**为例，验证了在消费级显卡上微调大语言模型做角色对话的可行性。

## 核心亮点

| 项目 | 数值 |
|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct（15 亿参数） |
| 微调方法 | QLoRA（4-bit NF4 量化 + LoRA r=8） |
| 训练数据 | LCCC-base 中文对话数据集（清洗后 510 万训练 / 57 万验证） |
| 可训练参数 | 920 万（仅占 0.59%） |
| Adapter 体积 | 35.3 MB |
| 训练耗时 | 11.6 小时（1 epoch，12,500 步） |
| 训练困惑度 PPL | 8.5 |
| 峰值显存 | ~3 GB / 8 GB |
| 硬件 | NVIDIA RTX 5060 Laptop（8 GB） |

## 功能特性

- **双模式切换**：DeepSeek API 模式（完整人设，回复质量高）与本地 QLoRA 模型模式（完全离线）。
- **情感分析模块**：基于 SnowNLP 的中文情绪三分类 + 关键词匹配，构成双轨好感度系统。
- **世界书引擎**：存储 23 条角色背景知识，通过关键词匹配动态注入上下文，保持角色一致性。
- **完整评估流水线**：自动化评估（`evaluate.py`）与人工评估（`manual_eval.py`）相结合。

## 技术架构

```
用户输入
  │
  ├─ 情感分析（SnowNLP + 关键词双轨）
  │     └─ 更新好感度
  │
  ├─ 世界书引擎（关键词匹配注入背景知识）
  │
  └─ 对话生成
        ├─ API 模式  → DeepSeek API（OpenAI 兼容）
        └─ 本地模式 → QLoRA 微调模型（4-bit 量化推理）
```

## 快速开始

环境要求：Windows 10/11、Python 3.13（本地模式需 8GB+ NVIDIA 显卡）。

```bash
pip install -r requirements.txt
```

### API 模式（推荐，回复质量最好）

在项目根目录创建 `.env` 文件并填入 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=你的Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

```bash
# PowerShell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python chat.py
```

### 本地模式（离线运行）

```bash
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:USE_LOCAL_MODEL = "true"
python chat.py
```

聊天内命令：`/help` 查看帮助、`/status` 查看角色状态、`/save` 保存、`/clear` 重置、`/exit` 退出。

## 目录结构

```
Al-chat/
├── chat.py              # 主程序入口（交互式聊天）
├── config.py            # 配置加载（API / 本地模型参数）
├── local_model.py       # 本地 QLoRA 模型加载与推理
├── sentiment.py         # 情感分析与好感度系统
├── miku_character.py    # 角色人设
├── world_book.py        # 世界书引擎
├── world_book.json      # 角色背景知识库（23 条）
├── evaluate.py          # 自动化评估
├── manual_eval.py       # 人工评估
├── train/               # 训练脚本（数据清洗 + QLoRA 微调）
├── docs/                # 文档
├── models/              # 模型权重（已 gitignore，需自行下载/训练）
└── data/                # 数据集（已 gitignore）
```

> 注：`models/`（约 3.1 GB 权重）与 `data/`（约 3.0 GB 数据集）未纳入版本控制，训练产物 LoRA adapter 仅 35 MB，可自行复现或联系作者获取。

## 训练流程

1. **数据清洗**：LCCC-base 数据集经 jieba 分词、停用词过滤、长度筛选、去重，得到 510 万训练 / 57 万验证样本。
2. **角色数据生成**：调用 DeepSeek API 生成 995 条三玖角色风格对话，注入人设。
3. **QLoRA 微调**：4-bit NF4 量化基座 + LoRA（r=8），峰值显存约 3 GB，12,500 步。

## 已知问题与展望

- **上下文窗口溢出**：训练时 `max_length` 为 512 token，完整角色设定（约 1000+ token）在本地模式下会溢出，导致回复退化为结构化元描述。已用精简人设规避，后续计划将 `max_length` 提升至 1024 重新训练。
- 更多细节见 `PROJECT_GUIDE.md`、`REPORT.md` 与 `期末设计报告_v4.docx`。

## 依赖

见 `requirements.txt`：`openai`、`python-dotenv`、`snownlp`、`jieba`、`torch`、`transformers`、`peft`、`bitsandbytes` 等。
