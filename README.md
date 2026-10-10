# 小说设定 RAG

把自己的小说设定做成**可问答的知识库**：设定切成片段 → 向量化 → 存入 Chroma → 提问时检索最相关片段 → 交给 DeepSeek 生成带来源引用的回答，并标出前后冲突。

> 一句话：**bge-m3 负责「找」，DeepSeek 负责「写」，你负责「裁决」。**

## 功能

- **多本小说隔离**：`data/<作品名>/` 下各自放 `.md` / `.txt`
- **问答 + 来源引用 + 冲突提示**：只依据检索到的片段回答，冲突处明确标出、由作者裁决
- **网页可视化**：展开查看命中的片段、来源、相似度、组装后的完整 prompt

## 使用案例

| 场景 | 问什么 | 结果 |
|------|--------|------|
| 查角色 | "林澈是谁？" | 检索角色卡，回答身份/年龄/师门，附【来源】 |
| 查世界观 | "这个世界有哪些境界？" | 检索世界观片段，列出练气/筑基/金丹/元婴 |
| 查时间线 | "林澈三年前发生了什么？" | 检索时间线片段 |
| **一致性检查（防吃书）** | "林澈几岁？" | 同时命中角色卡（17 岁）和旧稿（18 岁），**明确标出冲突**、把裁决权留给你 |

> 示例数据 `data/example/` 里故意放了一条冲突（`characters.md` 写 17 岁、`old_draft.md` 写 18 岁），问"林澈几岁"就能看到冲突检测效果。

## 快速开始（Windows）

1. 复制 `.env.example` 为 `.env`，填入你的 `DEEPSEEK_API_KEY`
2. 双击 `setup.bat`（一次性：建虚拟环境 + 装依赖，已用清华镜像加速）
3. 双击 `run_app.bat` 启动 → 浏览器打开 → 侧边栏「重建索引」→ 开始提问

> 第一次启动会下载 `bge-m3` 模型（约 2GB），耐心等一次，之后都是离线本地跑。

## 架构（数据流）

```
data/<作品名>/*.md|txt
   → loader（读文件）→ splitter（切段）→ embedder（bge-m3 向量化）
   → store（Chroma 向量库）
提问 → retriever（检索 top-k）→ generator（组装 prompt + DeepSeek 生成）
   → Streamlit 网页（含检索详情）
```

`bge-m3` 负责"找"（向量化/检索），`DeepSeek` 负责"写"（生成回答）。

## 目录结构

```
novel-rag/
  rag/            # 核心流水线（loader/splitter/embedder/store/retriever/generator/pipeline）
  app.py          # Streamlit 网页
  config.py       # 配置（读 .env）
  data/           # 你的设定放这里（每本小说一个子文件夹）
  tests/          # pytest 单元测试（13 个，用假实现，无需模型和 API key）
  setup.bat       # 一键装环境
  run_app.bat     # 一键启动
```

## 测试

```bash
python -m pytest -q
# 13 passed
```

单元测试用假实现（`FakeEmbedder` / `FakeGenerator`），不需要模型和 API key 也能跑。
