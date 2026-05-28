import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 支持 matplotlib 打印中文，防止图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# 1. 页面配置与暗黑系美化
# =========================================
st.set_page_config(
    page_title="冲击地压智能综合决策平台",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
.main-title { font-size: 40px; font-weight: bold; background: linear-gradient(90deg,#00F5A0,#00D9F5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px; }
.sub-title { font-size: 16px; color: #A0AEC0; margin-bottom: 25px; }
[data-testid="stChatMessage"] { background-color: #161B22; border-radius: 16px; padding: 15px; margin-bottom: 12px; border: 1px solid #30363D; }
section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #30363D; }
.stButton button { background: linear-gradient(90deg,#00F5A0,#00D9F5); color: black; border-radius: 10px; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)

# =========================================
# 2. 侧边栏导航控制
# =========================================
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:25px;">⛏️ 智能防冲平台</div>', unsafe_allow_html=True)
    st.divider()

    page_mode = st.radio(
        "📂 请选择功能模块：",
        ["📚 冲击地压文献知识库 (RAG)", "📊 钻进参数智能分析 (数据AI)"]
    )

    st.divider()
    st.header("⚙️ 系统状态")
    st.success("本地模型：Qwen2.5")
    st.success("前端响应：毫秒级秒上屏 ⚡")
    st.success("输出模式：流式打字机 🟢")

# =========================================
# 3. 核心性能优化：利用缓存加载知识库 (彻底解决卡顿)
# =========================================
pdf_folder = "docs"
DB_PATH = "vector_store"


@st.cache_resource  # 核心黑科技：让模型和论文只在启动时加载一次，回车再也不卡！
def load_knowledge_base():
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    # 如果已经有建好的数据库，直接秒级读取
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)

    # 如果没有，则去读 docs 文件夹里的 PDF
    if os.path.exists(pdf_folder) and any(f.endswith('.pdf') for f in os.listdir(pdf_folder)):
        documents = []
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                documents.extend(PyPDFLoader(os.path.join(pdf_folder, file)).load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embedding)
        db.save_local(DB_PATH)
        return db
    return None


# 在后台静默初始化数据库
db = load_knowledge_base()

# =========================================
# 4. 模块一：RAG 文献知识库 (秒上屏流式版)
# =========================================
if page_mode == "📚 冲击地压文献知识库 (RAG)":
    st.markdown('<div class="main-title">📚 冲击地压规程文献知识库</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">基于本地物理大模型 + 行业规程文献的智能问答系统</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📄 已加载参考规程")
        if os.path.exists(pdf_folder):
            for file in os.listdir(pdf_folder):
                if file.endswith(".pdf"):
                    st.markdown(f"- {file}")

    # 从 session_state 初始化或读取历史聊天记录（Streamlit 刷新不丢失的核心）
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 接收新问题
    if query := st.chat_input("请输入您想咨询的防冲问题..."):
        # 【核心改动】回车敲下瞬间，立刻把用户输入渲染在屏幕上！
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # 在下方为 AI 开辟一个流式输出的容器
        with st.chat_message("assistant"):
            response_placeholder = st.empty()  # 创建占位符

            # 检索知识库
            if db:
                results = db.similarity_search(query, k=3)
                context = "\n".join([doc.page_content for doc in results])
                source_text = ""
                for i, doc in enumerate(results):
                    source = doc.metadata.get("source", "未知来源")
                    page = doc.metadata.get("page", "未知页码")
                    source_text += f"\n[{i + 1}] {os.path.basename(source)} - 第 {page} 页"
                prompt = f"你是一名专业的冲击地压专家。请严格依据知识库内容回答问题。\n\n知识库：\n{context}\n\n问题：\n{query}"
            else:
                prompt = query
                source_text = "\n[提示] 未加载本地知识库，当前回答基于大模型自身知识。"

            # 开启 stream=True，让 Ollama 一个字一个字蹦出来
            full_response = ""
            stream = ollama.chat(
                model='qwen2.5',
                messages=[{'role': 'user', 'content': prompt}],
                stream=True
            )
            for chunk in stream:
                full_response += chunk['message']['content']
                response_placeholder.markdown(full_response + "▌")  # 动态打印

            # 打字结束后，去掉光标，拼上参考资料
            final_output = f"{full_response}\n\n---\n### 📖 参考来源\n{source_text}"
            response_placeholder.markdown(final_output)

        # 存入历史记录
        st.session_state.messages.append({"role": "assistant", "content": final_output})

# =========================================
# 5. 模块二：钻进参数智能分析 (保留 matplotlib 版)
# =========================================
elif page_mode == "📊 钻进参数智能分析 (数据AI)":
    st.markdown('<div class="main-title">📊 钻进参数时序智能分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">上传现场钻进动态数据，自动结合本地文献提供专家级防冲诊断</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("请上传钻进参数 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("数据上传成功！")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 数据预览 (前五行)")
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.subheader("📈 参数统计")
            st.dataframe(df.describe(), use_container_width=True)

        st.subheader("📉 参数变化曲线")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        selected_column = st.selectbox("选择参数", numeric_columns)

        # 绘制曲线
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(df[selected_column], color='#00D9F5', linewidth=1.5)
        ax.set_title(f"{selected_column} 参数随采样点变化趋势", color='white')
        ax.set_xlabel("采样点", color='white')
        ax.set_ylabel(selected_column, color='white')

        # 匹配暗黑主题美化 matplotlib
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#161B22')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        st.pyplot(fig)

        # AI 分析预备数据
        stats_text = df.describe().to_string()
        max_val = df[selected_column].max()
        mean_val = df[selected_column].mean()

        st.subheader("🤖 AI智能分析报告 (流式生成)")

        if st.button("🚀 开始 AI 专家系统分析"):
            report_placeholder = st.empty()
            full_report = ""

            # 自动联动第一功能模块里读进去的论文知识
            context = ""
            if db:
                results = db.similarity_search(f"钻进参数 {selected_column} 异常波动 冲击地压 预警指标", k=2)
                context = "\n".join([doc.page_content for doc in results])

            prompt = f"""
你是一名矿山冲击地压与钻进参数分析专家。请结合上传的现场监测数据和行业权威文献进行深度综合评估。

【钻进监测数据统计结果】
{stats_text}
关键特征：当前所选参数【{selected_column}】最大值为 {max_val:.2f}，平均值为 {mean_val:.2f}。

【参考行业文献规程（知识库提取）】
{context if context else "未检索到相关规程。"}

请从以下几个角度进行专业分析并输出报告：
1. 参数变化特征
2. 是否存在异常波动或突变
3. 是否可能存在高应力或冲击危险征兆
4. 具体的防冲工程建议
"""
            stream = ollama.chat(
                model='qwen2.5:1.5b',
                messages=[{'role': 'user', 'content': prompt}],
                stream=True
            )
            for chunk in stream:
                full_report += chunk['message']['content']
                report_placeholder.markdown(full_report + "▌")
            report_placeholder.markdown(full_report)