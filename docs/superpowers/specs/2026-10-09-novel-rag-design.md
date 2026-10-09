# 小说设定 RAG Agent — 设计文档

- 日期：2026-10-09
- 状态：已确认，待实现
- 作者：用户（小说作者）+ 助手

## 1. 目标

做一个能"查小说设定"的系统，最终升级成 agent。它同时服务于两个目的：

1. **练手**：一步步看清 RAG 全流程，以及 agent 是怎么在 RAG 之上"长"出来的。
2. **真实可用**：作者写文时查询设定、防吃书（发现前后冲突）。

核心能力：

- 把多本小说的设定（角色、地点、时间线、规则等）导入，建立可检索的向量库。
- 用户用自然语言提问 → 检索相关设定片段 → 交给 DeepSeek 生成有据可查的回答。
- 回答必须**引用来源**；若检索到的多条设定互相冲突，要**明确指出冲突**，由作者裁决，绝不自行改写设定。
- 全程**可视化**：能看到切分结果、检索命中了哪几段、组装后的 prompt。

## 2. 背景与约束

- 作者有数本小说的设定（单本约十几万字，尚未整理成文件）。
- 设定前后可能冲突——这是**特性**而非 bug：系统忠实呈现冲突，作者拍板。
- 技术栈已确定：
  - 语言：Python
  - 生成（LLM）：DeepSeek `deepseek-chat`
  - 向量化（embedding）：本地 `bge-m3`（DeepSeek 无 embedding 接口）
  - 向量库：Chroma（本地，不联网）
  - 切分/加载：LangChain（仅借用，核心流程手写，避免框架黑盒）
  - 网页：Streamlit
- 学习主线：朴素 RAG → 结构化 RAG → Agentic RAG。

## 3. 总体架构（数据流）

```
data/<作品名>/*.md|txt          ← 作者的设定文件
        │
        ▼
 ① 加载 + 切分（chunking）      ← 切成小段
        │
        ▼
 ② 向量化（bge-m3）             ← 每段 → 一串数字（坐标）
        │
        ▼
 ③ 存入向量库（Chroma）         ← 建索引
        │
        ▼
 ── 查询阶段 ──────────────────────────
 你提问
        │
        ▼
 ④ 问题向量化                   ← 问题 → 坐标
        │
        ▼
 ⑤ 检索 top-k                   ← 找最相近的几段
        │
        ▼
 ⑥ 组装 prompt + DeepSeek 生成  ← 问题 + 命中的段 → 回答
        │
        ▼
 ⑦ 网页展示：回答 + 命中的段（来源/分数）+ 完整 prompt
```

## 4. 组件设计

每个组件一个职责，接口清晰，可独立测试。

| 组件 | 职责 | 输入 → 输出 |
|------|------|-------------|
| `loader` | 读 `data/` 下文件 | 文件路径 → 文本 + 元数据（作品名、文件名） |
| `splitter` | 切分文本 | 文本 → chunk 列表 |
| `embedder` | 向量化（bge-m3） | 文本 → 向量 |
| `store` | 存取向量库（Chroma） | chunk+向量 → 索引；查询向量 → top-k chunk |
| `retriever` | 检索 | 问题 → top-k 命中的 chunk |
| `generator` | 组装 prompt + 调用 DeepSeek | 问题 + chunks → 回答 |
| `web` | Streamlit 网页 | 用户提问 → 回答 + 检索详情 |

依赖方向：`web → retriever → (embedder, store)`；`web → generator`。`retriever` 与 `generator` 相互独立。

## 5. 分阶段计划

### 阶段一（本次实现）：朴素 RAG
- 固定长度切分（LangChain `RecursiveCharacterTextSplitter`，带重叠）。
- top-k 检索（k=5）。
- 网页：聊天 + 每条回答可展开看命中的 chunk、来源、分数、完整 prompt。
- 用**示例设定**（一两段）跑通整条流水线。
- 验收：本地启动网页 → 提问 → 得到带来源的回答 + 能看检索详情。

### 阶段二（待真实材料后）：结构化 RAG
- 按设定类型（角色 / 地点 / 时间线 / 规则）切分 + 元数据（作品名、实体名、类型）。
- 多作品隔离：检索可限定在某部作品内。
- 具体切分规则等看到作者真实设定文件后再定。

### 阶段三：Agentic RAG
- 把"查角色 / 查时间线 / 查规则"封装成工具，模型自主决定调哪个、调几次。
- tool calling + 决策循环。
- 可选：把检索封装成 MCP server。

## 6. 项目结构

```
novel-rag/
  data/                # 作者设定放这里（每本小说一个子文件夹）
    example/           # 阶段一用的示例设定
  app.py               # Streamlit 网页入口
  rag.py               # 核心流水线（loader/splitter/embedder/store/retriever/generator）
  config.py            # 配置（DeepSeek API key 等）
  requirements.txt     # 依赖
  .env.example         # 配置模板（真实 .env 不提交）
  .gitignore           # 忽略 .env、向量库缓存、模型缓存
  README.md            # 给 GitHub 看的说明
  docs/superpowers/specs/   # 设计文档
```

## 7. 配置与密钥

- DeepSeek API key 通过环境变量 `.env` 提供，`config.py` 读取。
- `.env` 加入 `.gitignore`，绝不提交；提供 `.env.example` 模板。
- `bge-m3` 首次运行需联网下载权重，之后本地离线运行。

## 8. 错误处理与边界

- 未配置 API key → 网页给出清晰提示，不崩溃。
- 检索结果为空 → 明确回答"没找到相关设定"，**不瞎编**。
- 文件格式不支持 → 跳过并提示。
- 生成失败（网络/限流）→ 显示错误信息，不伪装成功。

## 9. 验收标准（阶段一 Definition of Done）

1. `pip install -r requirements.txt` 后能本地启动网页。
2. 导入 `data/example/` 示例设定，重建索引成功。
3. 提问得到回答，且回答附**来源引用**。
4. 网页能展开查看：命中的 chunk、来源文件、相似度分数、完整 prompt。
5. 代码结构清晰，能 `git push` 到 GitHub；README 完整。
