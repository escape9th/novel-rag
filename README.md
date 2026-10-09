# 小说设定 RAG（阶段一）

把自己的小说设定做成可问答的知识库：把设定切成片段、向量化、存入 Chroma，
提问时检索最相关片段，交给 DeepSeek 生成带来源引用的回答，并标出前后冲突。

## 功能

- **多本小说隔离**：`data/<作品名>/` 下各自放 `.md` / `.txt`
- **问答 + 来源引用 + 冲突提示**：只依据检索到的片段回答，冲突处明确标出、由作者裁决
- **网页可视化**：展开查看命中的片段、来源、相似度、组装后的完整 prompt

## 一键使用（Windows）

1. 复制 `.env.example` 为 `.env`，填入你的 `DEEPSEEK_API_KEY`
2. 双击 `setup.bat`（第一次运行，自动建环境 + 装依赖，较久）
3. 双击 `run_app.bat` 启动网页 → 点侧边栏「重建索引」→ 开始提问

> 以后每次用，只需要双击 `run_app.bat`（setup.bat 只需第一次跑）。

## 架构（数据流）

```
data/<作品名>/*.md|txt
   → loader（读文件）→ splitter（切段）→ embedder（bge-m3 向量化）
   → store（Chroma 向量库）
提问 → retriever（检索 top-k）→ generator（组装 prompt + DeepSeek 生成）
   → Streamlit 网页展示（含检索详情）
```

`bge-m3` 负责"找"（向量化/检索），`DeepSeek` 负责"写"（生成回答）。

## 手动跑测试

```bash
python -m pytest -q
```

单元测试用假实现（FakeEmbedder / FakeGenerator），不需要模型和 API key 也能跑。

## 目录结构

```
novel-rag/
  rag/            # 核心流水线（loader/splitter/embedder/store/retriever/generator/pipeline）
  app.py          # Streamlit 网页
  config.py       # 配置（读 .env）
  data/           # 你的设定放这里
  tests/          # pytest 单元测试
  setup.bat       # 一键装环境
  run_app.bat     # 一键启动
```
