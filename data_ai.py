import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ollama


# =========================================
# 页面配置
# =========================================

st.set_page_config(
    page_title="矿山工程数据智能分析平台",
    page_icon="⛏️",
    layout="wide"
)


# =========================================
# 页面美化
# =========================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

.main-title {
    font-size: 40px;
    font-weight: bold;
    background: linear-gradient(90deg,#00F5A0,#00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    color: #A0AEC0;
    margin-bottom: 30px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# 标题
# =========================================

st.markdown(
    '<div class="main-title">⛏️ 矿山工程数据智能分析平台</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">基于AI的钻进参数智能分析系统</div>',
    unsafe_allow_html=True
)


# =========================================
# 上传CSV
# =========================================

uploaded_file = st.file_uploader(
    "上传钻进参数CSV文件",
    type=["csv"]
)


# =========================================
# 数据分析
# =========================================

if uploaded_file is not None:

    # ===== 读取CSV =====

    df = pd.read_csv(uploaded_file)

    st.success("CSV文件上传成功！")

    # ===== 展示数据 =====

    st.subheader("📊 数据预览")

    st.dataframe(df.head())

    # ===== 基本统计 =====

    st.subheader("📈 参数统计")

    st.dataframe(df.describe())

    # =========================================
    # 绘制曲线
    # =========================================

    st.subheader("📉 参数变化曲线")

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    selected_column = st.selectbox(
        "选择参数",
        numeric_columns
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df[selected_column])

    ax.set_title(f"{selected_column} 参数变化")

    ax.set_xlabel("采样点")

    ax.set_ylabel(selected_column)

    st.pyplot(fig)

    # =========================================
    # AI分析
    # =========================================

    st.subheader("🤖 AI智能分析")

    # ===== 提取统计信息 =====

    stats_text = df.describe().to_string()

    # ===== Prompt =====

    prompt = f"""
你是一名矿山冲击地压与钻进参数分析专家。

下面是钻进监测数据统计结果：

{stats_text}

请从以下角度进行专业分析：

1. 参数变化特征
2. 是否存在异常波动
3. 是否可能存在高应力区域
4. 是否存在冲击危险征兆
5. 给出工程建议

请使用专业、详细、工程化的语言回答。
"""

    # ===== AI分析按钮 =====

    if st.button("开始AI分析"):

        with st.spinner("AI正在分析工程数据..."):

            response = ollama.chat(
                model='qwen2.5:1.5b',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )

            answer = response['message']['content']

            st.markdown(answer)