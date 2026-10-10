# NovelRAG · 小说设定 RAG 知识库 / Novel Settings RAG Knowledge Base

> **bge-m3 负责「找」，DeepSeek 负责「写」，作者负责「裁决」。**
> *bge-m3 finds, DeepSeek writes, the author decides.*

把你的小说设定（**角色 / 世界观 / 时间线 / 规则**）做成一个**可问答的知识库**。用 **RAG（Retrieval-Augmented Generation，检索增强生成）** 把设定切成片段、向量化、存入**向量数据库**，提问时检索最相关的片段，交给 **LLM（DeepSeek）** 生成**带来源引用**的回答，并**主动标出设定冲突**（防吃书）。

Turn your novel/fiction settings (**characters / worldbuilding / timeline / rules**) into a queryable **knowledge base**. Using **RAG (Retrieval-Augmented Generation)**, it chunks your settings, embeds them into a **vector database**, retrieves the most relevant chunks on question, and lets an **LLM (DeepSeek)** answer with **source citations** and **conflict detection** — so you never break your own canon.

---

## 🔑 Keywords / 关键词

`RAG` · `Retrieval-Augmented Generation` · `AI Agent` · `LLM` · `LangChain` · `Chroma` · `Vector Database` · `Text Embedding` · `bge-m3` · `sentence-transformers` · `DeepSeek` · `Semantic Search` · `Knowledge Base` · `Novel Writing` · `Worldbuilding` · `设定管理` · `一致性检查` · `防吃书` · `Tool Calling` · `MCP` · `Streamlit` · `Python`

---

## ✨ Features / 功能

- **多本小说隔离 / Multi-novel isolation** — `data/<作品名>/` 下各自放 `.md` / `.txt`，检索互不污染
- **问答 + 来源引用 / Q&A with source citation** — 只依据检索到的片段回答，不编造
- **冲突检测 / Conflict detection** — 多条设定互相矛盾时明确标出，把裁决权留给作者（**防吃书**）
- **网页可视化 / Visual web UI** — 展开查看命中的片段、来源文件、相似度分数、组装后的完整 prompt
- **本地离线向量化 / Offline embedding** — `bge-m3` 本地跑，免费、无 API 调用

## 🎯 Use Cases / 使用案例

| 场景 Scenario | 问题 Question | 结果 Result |
|------|------|------|
| 查角色 Character | "林澈是谁？" | 检索角色卡，回答身份/年龄/师门，附【来源】 |
| 查世界观 Worldbuilding | "这个世界有哪些境界？" | 列出练气 / 筑基 / 金丹 / 元婴 |
| 查时间线 Timeline | "林澈三年前发生了什么？" | 检索时间线片段 |
| **一致性检查 Consistency check** | "林澈几岁？" | 同时命中角色卡（17 岁）与旧稿（18 岁），**明确标出冲突** |

> 示例数据 `data/example/` 里故意放了一条冲突（`characters.md` 写 17 岁、`old_draft.md` 写 18 岁），问「林澈几岁」即可看到冲突检测效果。

## 🏗 Architecture / 架构（数据流）

```
【建索引 / Indexing】
data/<作品名>/*.md|txt
   → loader     读文件 (read files)
   → splitter   切段   (chunk text)
   → embedder   向量化 (embed with bge-m3)
   → store      存向量库 (Chroma vector database)

【问答 / Query】
提问 question
   → embedder   问题向量化
   → retriever  检索 top-k (semantic search)
   → generator  组装 prompt + DeepSeek 生成 (LLM generation)
   → Streamlit  网页展示（含检索详情 / retrieval details）
```

## 🧰 Tech Stack / 技术栈

| 层 Layer | 技术 Tech | 用途 Purpose |
|------|------|------|
| Language | Python ≥ 3.11 | 主体（别用 3.14，torch 支持慢半拍） |
| Embedding | `sentence-transformers` + `bge-m3` | 本地向量化（免费离线） |
| Vector DB | `Chroma` | 本地向量存储 + 相似度检索 |
| Chunking | `langchain-text-splitters` | 递归切分（核心流程手写，不被框架黑盒） |
| LLM | `DeepSeek` (OpenAI-compatible API) | 生成回答 |
| Web UI | `Streamlit` | 聊天界面 + 检索可视化 |
| Testing | `pytest` | 13 个单元测试，用假实现无需 key |

> **设计原则**：LangChain 只用于切分，RAG 核心流程（加载→切分→向量化→检索→生成）全部手写，便于学习与掌控。

## 🚀 Quick Start / 快速开始（Windows）

1. 复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`
   *Copy `.env.example` to `.env` and fill in `DEEPSEEK_API_KEY`*
2. 双击 `setup.bat`（一次性：建虚拟环境 + 装依赖，已用清华镜像加速）
   *Double-click `setup.bat` (one-time: create venv + install deps)*
3. 双击 `run_app.bat` → 浏览器打开 → 侧边栏「重建索引」→ 提问
   *Double-click `run_app.bat` → open browser → "重建索引" in sidebar → ask*

> 第一次启动会下载 `bge-m3` 模型（约 2GB），之后离线本地运行。
> *First run downloads the `bge-m3` model (~2GB); afterwards it runs fully offline.*

## 📁 Project Structure / 目录结构

```
novel-rag/
  rag/            # 核心流水线 core pipeline (loader/splitter/embedder/store/retriever/generator/pipeline)
  app.py          # Streamlit 网页 web UI
  config.py       # 配置 config (.env)
  data/           # 你的设定放这里 put your settings here (每本小说一个子文件夹 one folder per novel)
  tests/          # pytest 单元测试 (13 tests, 无需模型和 API key)
  setup.bat       # 一键装环境 one-click env setup
  run_app.bat     # 一键启动 one-click launch
```

## 🧪 Tests / 测试

```bash
python -m pytest -q
# 13 passed
```

单元测试用假实现（`FakeEmbedder` / `FakeGenerator`），不需要模型和 API key 也能跑。
*Tests use fakes (`FakeEmbedder` / `FakeGenerator`), so they run without any model or API key.*

## 🗺 Roadmap / 路线图

- [ ] **阶段二 · 结构化 RAG**：按设定类型（角色/地点/时间线/规则）切分 + 元数据，检索更精准
- [ ] **阶段三 · Agent 化**：检索封装为工具，模型自主决定调用（tool calling + 决策循环）
- [ ] **阶段四 · MCP**：把检索封装成 MCP server，供其他 AI 应用调用
- [ ] 混合检索（向量 + BM25）、reranker 重排、跨文件实体链接
