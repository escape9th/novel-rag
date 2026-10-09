import streamlit as st

import config
from rag.embedder import BgeM3Embedder
from rag.generator import Generator, build_messages
from rag.pipeline import Pipeline
from rag.retriever import Retriever
from rag.store import VectorStore

st.set_page_config(page_title="小说设定 RAG", page_icon="📚", layout="wide")


@st.cache_resource  # 模型加载很慢，缓存起来只加载一次
def get_pipeline():
    embedder = BgeM3Embedder(config.EMBED_MODEL_NAME)
    store = VectorStore(config.CHROMA_DIR)
    retriever = Retriever(embedder, store, config.TOP_K)
    generator = Generator(config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL)
    return Pipeline(config.DATA_DIR, config.CHROMA_DIR, embedder, store, retriever,
                    generator, config.CHUNK_SIZE, config.CHUNK_OVERLAP)


pipe = get_pipeline()

st.title("📚 小说设定 RAG")
st.caption("基于你的设定库问答，回答会引用来源并标出冲突。")

with st.sidebar:
    st.header("索引管理")
    if st.button("重建索引"):
        with st.spinner("正在读取设定 → 切分 → 向量化…"):
            n = pipe.build_index()
        st.success(f"已建立 {n} 个片段")
    st.info(f"数据目录：`{config.DATA_DIR}`")
    if not config.DEEPSEEK_API_KEY:
        st.warning("未检测到 DEEPSEEK_API_KEY，请在 `.env` 中配置后重启。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("问一个关于你小说设定的问题…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            answer, hits = pipe.query(question)
        st.markdown(answer)

        with st.expander("🔍 查看检索详情"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**片段{i}** · 来源 `{h.work}/{h.source}` · 相似度 `{h.score:.3f}`")
                st.text(h.text)

        with st.expander("🧠 查看组装后的 prompt"):
            for m in build_messages(question, hits):
                st.markdown(f"**{m['role']}**")
                st.code(m["content"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
