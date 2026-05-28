# -*- coding: utf-8 -*-
"""
DeepRB: 基于本地 Qwen2.5 与知识增强的冲击地压领域专家助手。

功能：
1. 冲击地压知识问答：RAG 检索、引用来源、证据不足时拒绝过度判断。
2. 随钻参数分析：兼容 Excel/CSV/TXT，输出 MSE 近似反演、应力集中系数与风险提示。
3. 技术日报生成：基于上一页分析结果与知识库证据生成结构化日报。
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import ollama
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "docs"
DB_PATH = BASE_DIR / "vector_store"
EXPERT_PROMPT_PATH = BASE_DIR / "rockburst_expert_prompt.md"
SECOND_FEATURE_DIR = BASE_DIR / "second_feature"
DEFAULT_MODEL = "qwen2.5"
PDF_COUNT = len(list(PDF_FOLDER.glob("*.pdf"))) if PDF_FOLDER.exists() else 0

for folder in [PDF_FOLDER, DB_PATH]:
    folder.mkdir(exist_ok=True)


CRITICAL_KNOWLEDGE_BASE = {
    "冲击倾向性基本指标": (
        "动态破坏时间、弹性能指数、冲击能指数、单轴抗压强度等指标可用于描述煤岩冲击倾向性；"
        "具体阈值必须以适用标准、试验规程或矿井防冲专项设计为准。"
    ),
    "应力集中判据": (
        "本系统演示判据：应力集中系数 K = 反演应力 sigma / 煤体单轴抗压强度 sigma_c。"
        "当 K >= 1.5 时标记为高关注区。该阈值仅用于项目演示，现场应用必须经规程和专家复核。"
    ),
    "解危原则": (
        "冲击危险区应结合区域卸压、局部卸压、监测预警和效果检验综合处置；"
        "钻孔直径、孔深、孔距、爆破参数等不得脱离煤层赋存、巷道布置、地应力和现场规程直接套用。"
    ),
}


HIGH_RISK_KEYWORDS = [
    "撤人",
    "停产",
    "强冲击",
    "危险等级",
    "能不能生产",
    "是否安全",
    "卸压参数",
    "爆破",
    "钻孔直径",
    "孔距",
    "孔深",
]


st.set_page_config(
    page_title="DeepRB - 冲击地压领域专家助手",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0E1117; color: #F8FAFC; }
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #E5F4FF;
        margin-bottom: 4px;
    }
    .sub-title { font-size: 14px; color: #A7B0BE; margin-bottom: 24px; }
    [data-testid="stChatMessage"] {
        background-color: #151A22;
        border: 1px solid #2B3340;
        border-radius: 8px;
        padding: 14px;
    }
    [data-testid="stSidebar"] {
        background-color: #101722;
        border-right: 1px solid #2B3340;
    }
    .stButton button {
        background-color: #00D9F5;
        color: #061016;
        font-weight: 700;
        border-radius: 8px;
        border: none;
    }
    .evidence-box {
        border: 1px solid #2B3340;
        border-radius: 8px;
        padding: 12px;
        background: #111822;
        margin-bottom: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_expert_prompt() -> str:
    if EXPERT_PROMPT_PATH.exists():
        return EXPERT_PROMPT_PATH.read_text(encoding="utf-8")

    return (
        "你是 DeepRB，一个严谨的冲击地压领域专家助手。回答必须基于证据，"
        "区分文献依据、模型推断和需要现场复核的信息；证据不足时不得编造结论。"
    )


@st.cache_resource(show_spinner="首次检索正在加载知识库，请稍候...")
def load_rag_db():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        if (DB_PATH / "index.faiss").exists() and (DB_PATH / "index.pkl").exists():
            return FAISS.load_local(
                str(DB_PATH),
                embedding,
                allow_dangerous_deserialization=True,
            )

        pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
        if not pdf_files:
            return None

        documents = []
        for pdf_file in pdf_files:
            documents.extend(PyPDFLoader(str(pdf_file)).load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=180,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )
        chunks = splitter.split_documents(documents)
        db = FAISS.from_documents(chunks, embedding)
        db.save_local(str(DB_PATH))
        return db
    except Exception as exc:
        st.sidebar.warning(f"知识库加载失败：{exc}")
        return None


def get_rag_db():
    """Lazy-load the vector database only when a task really needs retrieval."""
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True

    if not st.session_state.rag_enabled:
        return None

    return load_rag_db()


@st.cache_resource(show_spinner="正在初始化随钻反演工程内核...")
def init_drilling_kernels():
    """Load the imported second feature only when the second module is used."""
    second_feature_path = str(SECOND_FEATURE_DIR)
    if second_feature_path not in sys.path:
        sys.path.insert(0, second_feature_path)

    from core.coal_mechanics_inversion import CoalMechanicsInversion
    from core.data_acquisition import DataAcquisitionModule
    from core.signal_denoising import SignalDenoisingModule
    from utils.config import ConfigManager
    from utils.logger import LoggerManager

    cfg = ConfigManager()
    log = LoggerManager(cfg)
    return {
        "acq": DataAcquisitionModule(cfg, log),
        "den": SignalDenoisingModule(cfg, log),
        "inv": CoalMechanicsInversion(cfg, log),
    }


def save_uploaded_drilling_file(uploaded_file) -> Path:
    target_dir = SECOND_FEATURE_DIR / "raw_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target_path = target_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def enrich_drilling_result(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    sigma = result["coal_in_situ_stress_final_sigma"].clip(lower=0)
    sigma_c = result["coal_uniaxial_compressive_strength_sigma_c"].clip(lower=0.01)
    result["stress_concentration_k"] = sigma / sigma_c
    result["risk_flag"] = np.select(
        [result["stress_concentration_k"] >= 1.5, result["stress_concentration_k"] >= 1.2],
        ["高关注", "需跟踪"],
        default="常规",
    )
    return result


def run_drilling_project_pipeline(
    source_path: Path,
    h_ref: float,
    gamma: float,
    k_side: float,
    wave_level: int,
    kalman_q: float,
) -> pd.DataFrame:
    kernels = init_drilling_kernels()
    raw_df = kernels["acq"].acquire_data(str(source_path))
    denoised_df = kernels["den"].denoise_all_sensors(raw_df, wave_level=wave_level, q_val=kalman_q)
    result_df = kernels["inv"].invert_all_points(denoised_df, h0=h_ref, gamma=gamma, k_side=k_side)
    return enrich_drilling_result(result_df)


def stream_ollama(prompt: str, model: str, placeholder) -> str:
    answer = ""
    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            answer += chunk.get("message", {}).get("content", "")
            placeholder.markdown(answer + "▌")
        placeholder.markdown(answer)
    except Exception as exc:
        answer = f"模型调用失败：{exc}\n\n请确认 Ollama 已启动，并且本地模型 `{model}` 已安装。"
        placeholder.error(answer)
    return answer


def is_high_risk_question(query: str) -> bool:
    return any(keyword in query for keyword in HIGH_RISK_KEYWORDS)


def retrieve_evidence(db, query: str, k: int = 5) -> list[dict]:
    if db is None:
        return []

    try:
        docs_with_scores = db.similarity_search_with_score(query, k=k)
        evidence = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata or {}
            source = os.path.basename(str(metadata.get("source", "未知来源")))
            page = metadata.get("page", "未知")
            page_label = int(page) + 1 if isinstance(page, int) else page
            evidence.append(
                {
                    "source": source,
                    "page": page_label,
                    "score": float(score) if isinstance(score, (int, float, np.floating)) else None,
                    "content": doc.page_content.strip().replace("\x00", " "),
                }
            )
        return evidence
    except Exception:
        docs = db.similarity_search(query, k=k)
        return [
            {
                "source": os.path.basename(str((doc.metadata or {}).get("source", "未知来源"))),
                "page": (doc.metadata or {}).get("page", "未知"),
                "score": None,
                "content": doc.page_content.strip().replace("\x00", " "),
            }
            for doc in docs
        ]


def evidence_to_context(evidence: Iterable[dict], max_chars: int = 4200) -> str:
    parts = []
    used = 0
    for idx, item in enumerate(evidence, start=1):
        content = item["content"][:900]
        block = f"[证据{idx}] 来源：{item['source']}，页码：{item['page']}\n{content}"
        used += len(block)
        if used > max_chars:
            break
        parts.append(block)
    return "\n\n".join(parts)


def render_evidence(evidence: list[dict]) -> None:
    if not evidence:
        st.info("本次没有检索到可引用证据。")
        return

    with st.expander("查看本次检索证据", expanded=False):
        for idx, item in enumerate(evidence, start=1):
            score_text = f"，距离分数：{item['score']:.4f}" if item["score"] is not None else ""
            st.markdown(
                f"""
<div class="evidence-box">
<b>[{idx}] {item['source']}</b>，页码：{item['page']}{score_text}<br>
{item['content'][:260]}...
</div>
""",
                unsafe_allow_html=True,
            )


def build_qa_prompt(query: str, evidence: list[dict]) -> str:
    expert_prompt = load_expert_prompt()
    context = evidence_to_context(evidence)
    high_risk_notice = ""
    if is_high_risk_question(query):
        high_risk_notice = (
            "\n\n【高风险问题提示】用户问题涉及现场安全或施工参数。"
            "必须先判断证据是否充分；证据不足时不得给出最终处置结论，"
            "只能给出需补充信息、临时安全核查建议和复核要求。"
        )

    if not context:
        context = "本次未检索到可靠证据。请严格说明依据不足，不得编造来源、页码、阈值或标准条文。"

    return f"""{expert_prompt}

【项目内置规则，仅作辅助，不能替代现场规程】
{CRITICAL_KNOWLEDGE_BASE['冲击倾向性基本指标']}
{CRITICAL_KNOWLEDGE_BASE['应力集中判据']}
{CRITICAL_KNOWLEDGE_BASE['解危原则']}
{high_risk_notice}

【检索证据】
{context}

【用户问题】
{query}

请按以下结构回答：
### 结论
### 依据
### 专业分析
### 工程建议
### 需补充信息
### 引用来源
"""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "时间": ["时间", "time", "Time", "采样时间"],
        "推力": ["推力", "推进力", "轴压", "thrust", "Thrust", "feed_force"],
        "扭矩": ["扭矩", "转矩", "torque", "Torque"],
        "转速": ["转速", "旋转速度", "rpm", "RPM", "rot_speed"],
        "位移": ["位移", "深度", "孔深", "depth", "Depth", "displacement"],
    }

    rename = {}
    for standard_name, aliases in alias_map.items():
        for column in df.columns:
            if str(column).strip() in aliases:
                rename[column] = standard_name
                break
    return df.rename(columns=rename)


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    for encoding in ["utf-8-sig", "gbk", "gb18030", "utf-16"]:
        try:
            return pd.read_csv(io.BytesIO(raw), header=None, encoding=encoding, on_bad_lines="skip")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw), header=None, encoding_errors="ignore", on_bad_lines="skip")


def parse_and_clean_mining_file(uploaded_file) -> pd.DataFrame | None:
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded_file, header=None)
        else:
            df_raw = read_csv_flexible(uploaded_file)

        header_row = None
        for idx, row in df_raw.iterrows():
            row_text = " ".join(str(x) for x in row.values if pd.notna(x))
            if any(key in row_text for key in ["推力", "推进力", "thrust", "Thrust"]) and any(
                key in row_text for key in ["扭矩", "转矩", "torque", "Torque"]
            ):
                header_row = idx
                break

        uploaded_file.seek(0)
        if header_row is None:
            if file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                raw = uploaded_file.read()
                uploaded_file.seek(0)
                for encoding in ["utf-8-sig", "gbk", "gb18030"]:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=encoding, on_bad_lines="skip")
                        break
                    except Exception:
                        df = None
                if df is None:
                    st.error("无法识别 CSV 编码，请另存为 UTF-8 或 GBK 后重试。")
                    return None
        else:
            if file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file, skiprows=header_row)
            else:
                raw = uploaded_file.read()
                df = None
                for encoding in ["utf-8-sig", "gbk", "gb18030"]:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), skiprows=header_row, encoding=encoding, on_bad_lines="skip")
                        break
                    except Exception:
                        continue
                if df is None:
                    st.error("无法读取 CSV/TXT 数据，请检查文件格式。")
                    return None

        df = normalize_columns(df)
        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

        required = ["推力", "扭矩", "转速"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            st.error(f"数据缺少必要列：{', '.join(missing)}。至少需要推力、扭矩、转速。")
            return None

        for column in ["时间", "推力", "扭矩", "转速", "位移"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=required).copy()
        if df.empty:
            st.error("清洗后没有有效数值行，请检查数据列是否为数字。")
            return None

        if "位移" in df.columns and df["位移"].notna().sum() >= 5:
            df["depth"] = df["位移"].interpolate().ffill().bfill()
        elif "时间" in df.columns and df["时间"].notna().sum() >= 5:
            df["depth"] = np.linspace(0, max(len(df) * 0.2, 1), len(df))
        else:
            df["depth"] = np.linspace(0, 30, len(df))

        return df.sort_values("depth").reset_index(drop=True)
    except Exception as exc:
        st.error(f"数据解析失败：{exc}")
        return None


def make_demo_data() -> pd.DataFrame:
    depths = np.linspace(0, 30, 150)
    anomaly = np.where((depths >= 15) & (depths <= 22), 1.0, 0.0)
    return pd.DataFrame(
        {
            "depth": depths,
            "时间": depths * 2,
            "推力": 14 + np.random.normal(0, 0.4, len(depths)),
            "扭矩": 180 + np.sin(depths / 2.0) * 30 + anomaly * 120 + np.random.normal(0, 5, len(depths)),
            "转速": 280 + np.random.normal(0, 2, len(depths)),
        }
    )


def analyze_drilling_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    depths = raw_df["depth"].to_numpy(dtype=float)
    thrusts = raw_df["推力"].to_numpy(dtype=float)
    torques = raw_df["扭矩"].to_numpy(dtype=float)
    rot_speeds = raw_df["转速"].to_numpy(dtype=float)

    penetration_rates = 1.6 - (torques > np.nanpercentile(torques, 80)).astype(int) * 0.5
    penetration_rates += np.random.normal(0, 0.04, len(depths))
    penetration_rates = np.clip(penetration_rates, 0.2, 2.0)

    bit_diameter = 0.042
    area = np.pi * (bit_diameter**2) / 4
    mse = (thrusts * 1000 / area) + (
        48 * np.pi * rot_speeds * torques / ((bit_diameter**2) * (penetration_rates / 60))
    )
    mse_mpa = mse / 1e6

    base_strength = 16.0 + np.sin(depths / 4.0) * 2.0
    simulated_stress = 10.0 + depths * 0.15 + mse_mpa * 0.015
    stress_ratio = simulated_stress / base_strength
    plastic_radius = 1.0 + stress_ratio * 0.35

    result = pd.DataFrame(
        {
            "depth": depths,
            "mse_mpa": mse_mpa,
            "coal_uniaxial_compressive_strength_sigma_c": base_strength,
            "coal_in_situ_stress_final_sigma": simulated_stress,
            "stress_concentration_k": stress_ratio,
            "plastic_zone_radius_rp": plastic_radius,
        }
    )
    result["risk_flag"] = np.select(
        [result["stress_concentration_k"] >= 1.5, result["stress_concentration_k"] >= 1.2],
        ["高关注", "需跟踪"],
        default="常规",
    )
    return result


def build_report_prompt(res_df: pd.DataFrame, evidence: list[dict]) -> str:
    avg_strength = res_df["coal_uniaxial_compressive_strength_sigma_c"].mean()
    max_stress = res_df["coal_in_situ_stress_final_sigma"].max()
    max_k = res_df["stress_concentration_k"].max()
    max_rp = res_df["plastic_zone_radius_rp"].max()
    high_zone = res_df[res_df["stress_concentration_k"] >= 1.5]
    if high_zone.empty:
        zone_text = "未发现 K >= 1.5 的连续高关注区。"
    else:
        zone_text = f"{high_zone['depth'].min():.1f} m 至 {high_zone['depth'].max():.1f} m"

    return f"""{load_expert_prompt()}

【随钻反演结果】
- 煤体单轴抗压强度均值 sigma_c：{avg_strength:.2f} MPa
- 反演应力最大值 sigma_max：{max_stress:.2f} MPa
- 最大应力集中系数 K：{max_k:.2f}
- 最大塑性区半径 r_p：{max_rp:.2f} m
- 高关注深度段：{zone_text}

【内置判据说明】
{CRITICAL_KNOWLEDGE_BASE['应力集中判据']}
{CRITICAL_KNOWLEDGE_BASE['解危原则']}

【检索证据】
{evidence_to_context(evidence) or '本次未检索到足够可靠证据。'}

请生成一份“冲击地压随钻反演与防冲诊断技术日报”，必须包含：
1. 今日监测概况
2. 随钻物理比能与应力集中分析
3. 风险研判，明确哪些属于证据、哪些属于模型推断
4. 现场处置与复核建议
5. 后续重点监测指标
6. 引用来源

注意：不得把本演示判据直接表述为国家标准；涉及施工参数时必须要求现场规程和专家复核。
"""


with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:22px;">DeepRB 冲击地压专家助手</div>', unsafe_allow_html=True)
    st.caption("本地 Qwen2.5 + 冲击地压知识增强")
    st.divider()

    model_name = st.text_input("Ollama 模型名称", value=DEFAULT_MODEL)
    enable_rag = st.toggle("启用知识库检索", value=True, help="关闭后页面和回答会更快，但不引用本地论文。")
    st.session_state.rag_enabled = enable_rag
    agent_mode = st.radio(
        "选择功能模块",
        [
            "知识问答",
            "随钻载荷解析与力学反演",
            "预警日报生成",
        ],
    )

    st.divider()
    st.markdown("### 系统状态")
    st.success(f"模型：{model_name}")
    if enable_rag:
        st.info("知识库：按需加载，首次检索时启动")
    else:
        st.warning("知识库：已关闭")
    st.caption(f"论文数量：{PDF_COUNT}")
    st.caption(f"第二功能工程：{'已接入' if SECOND_FEATURE_DIR.exists() else '未找到'}")


if agent_mode == "知识问答":
    st.markdown('<div class="main-title">冲击地压知识问答</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">回答会优先基于本地论文和规则，并显式区分依据、推断与需复核信息。</div>',
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("请输入冲击地压机理、监测预警、防治措施或规程相关问题...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_messages.append({"role": "user", "content": query})

        db = get_rag_db()
        evidence = retrieve_evidence(db, query, k=5)
        prompt = build_qa_prompt(query, evidence)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer = stream_ollama(prompt, model_name, placeholder)
            render_evidence(evidence)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


elif agent_mode == "随钻载荷解析与力学反演":
    st.markdown('<div class="main-title">随钻载荷解析与力学反演</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">已整合压缩包工程：数据采集标准化、小波-卡尔曼降噪、MSE煤体力学反演和结果导出。</div>',
        unsafe_allow_html=True,
    )

    if not SECOND_FEATURE_DIR.exists():
        st.error("未找到 second_feature 工程目录，请确认压缩包业务代码已解压。")
        st.stop()

    with st.sidebar:
        st.divider()
        st.markdown("### 第二功能参数")
        h_ref = st.number_input("埋藏深度 H (m)", min_value=1.0, value=600.0, step=10.0)
        gamma_eff = st.number_input("覆盖层平均容重 γ (kN/m³)", min_value=1.0, value=24.5, step=0.5)
        k_lat = st.slider("侧向应力系数", 0.5, 3.0, 1.2, 0.1)
        wave_level = st.slider("小波分解阶数", 2, 6, 5)
        smooth_level = st.select_slider("卡尔曼平滑强度", options=["弱", "中", "强"], value="中")
        kalman_q = {"弱": 0.1, "中": 0.01, "强": 0.001}[smooth_level]

    uploaded_file = st.file_uploader("导入随钻原始记录文件", type=["xlsx", "xls", "csv"])
    sample_path = SECOND_FEATURE_DIR / "raw_data" / "418.1(1).xlsx"

    c_run, c_sample = st.columns([1, 2])
    with c_run:
        run_clicked = st.button("开始全流程分析", type="primary")
    with c_sample:
        use_sample = st.checkbox("未上传时使用工程自带示例数据", value=True)

    if run_clicked:
        try:
            if uploaded_file is not None:
                source_path = save_uploaded_drilling_file(uploaded_file)
            elif use_sample and sample_path.exists():
                source_path = sample_path
            else:
                st.error("请上传 Excel/CSV 数据，或勾选使用工程自带示例数据。")
                st.stop()

            progress = st.progress(0)
            status = st.empty()
            status.info("正在调用第二功能工程内核...")
            progress.progress(20)

            res_df = run_drilling_project_pipeline(
                source_path=source_path,
                h_ref=h_ref,
                gamma=gamma_eff,
                k_side=k_lat,
                wave_level=wave_level,
                kalman_q=kalman_q,
            )
            progress.progress(100)
            status.success("随钻载荷解析、降噪与煤体力学反演已完成。")

            st.session_state.current_res_df = res_df
            st.session_state.second_feature_source = str(source_path)
        except ModuleNotFoundError as exc:
            st.error(f"第二功能缺少依赖：{exc.name}。请先执行 `pip install -r requirements.txt`。")
        except Exception as exc:
            st.error(f"第二功能分析失败：{exc}")

    if "current_res_df" in st.session_state:
        res_df = st.session_state.current_res_df
        st.caption(f"当前结果来源：{st.session_state.get('second_feature_source', '会话内数据')}")

        c1, c2, c3, c4 = st.columns(4)
        mse_col = "mse" if "mse" in res_df.columns else "mse_mpa"
        c1.metric("平均 MSE", f"{res_df[mse_col].mean():.2f} MPa")
        c2.metric("平均单轴强度", f"{res_df['coal_uniaxial_compressive_strength_sigma_c'].mean():.2f} MPa")
        c3.metric("最大 K 值", f"{res_df['stress_concentration_k'].max():.2f}")
        c4.metric("最大塑性区半径", f"{res_df['plastic_zone_radius_rp'].max():.3f} m")

        high_zone = res_df[res_df["stress_concentration_k"] >= 1.5]
        if high_zone.empty:
            st.success("按项目演示判据，当前结果未出现 K >= 1.5 的高关注区。")
        else:
            st.warning(
                f"按项目演示判据，{high_zone['depth'].min():.1f} m 至 "
                f"{high_zone['depth'].max():.1f} m 出现 K >= 1.5 的高关注区。"
            )
            st.caption("该结论为反演提示，需结合钻屑量、微震、地质构造、采掘扰动和矿井防冲设计复核。")

        t_curve, t_quality, t_data, t_note = st.tabs(["反演曲线", "降噪效果", "结果数据", "模块说明"])
        with t_curve:
            fig = px.line(
                res_df,
                x="depth",
                y=[
                    "coal_uniaxial_compressive_strength_sigma_c",
                    "coal_in_situ_stress_final_sigma",
                    "stress_concentration_k",
                    "plastic_zone_radius_rp",
                ],
                labels={"depth": "孔深 m", "value": "指标值", "variable": "反演参数"},
                title="煤体强度、原地应力、应力集中系数与塑性区半径随孔深演化",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        with t_quality:
            channels = {
                "钻压 WOB": "WOB",
                "扭矩 torque": "torque",
                "转速 rotational_speed": "rotational_speed",
                "钻速 ROP": "ROP",
            }
            available = {label: col for label, col in channels.items() if col in res_df.columns and f"{col}_raw" in res_df.columns}
            if not available:
                st.info("当前结果中没有可用于降噪前后对比的 raw 字段。")
            else:
                label = st.selectbox("选择观测物理量", list(available.keys()))
                col = available[label]
                compare_df = pd.DataFrame(
                    {
                        "采样点": np.arange(len(res_df)),
                        "原始序列": res_df[f"{col}_raw"],
                        "净化序列": res_df[col],
                    }
                )
                fig_q = px.line(
                    compare_df,
                    x="采样点",
                    y=["原始序列", "净化序列"],
                    title=f"{label} 降噪前后对比",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_q, use_container_width=True)

        with t_data:
            st.dataframe(res_df, use_container_width=True)
            csv = res_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载完整反演结果 CSV", csv, "DeepRB_随钻反演结果.csv", "text/csv")

        with t_note:
            st.markdown(
                """
第二功能来自你提供的随钻数据处理工程，当前已接入主平台工作流：

1. 数据采集模块自动识别钻压、扭矩、转速、深度等字段。
2. 信号处理模块执行 db4 小波降噪与卡尔曼平滑。
3. 力学反演模块基于修正 MSE 和 Mohr-Coulomb 关系计算煤体强度、原地应力和塑性区半径。
4. 反演结果会保存到当前会话，并作为第三功能“预警日报生成”的输入。

注意：K >= 1.5 是本平台演示判据，不应直接表述为国家标准或现场最终判据。
"""
            )


elif agent_mode == "预警日报生成":
    st.markdown('<div class="main-title">预警日报生成</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">基于随钻分析结果、知识库证据和专家提示词生成结构化日报。</div>',
        unsafe_allow_html=True,
    )

    if "current_res_df" not in st.session_state:
        st.warning("请先进入“随钻载荷解析与力学反演”模块上传数据或运行示例数据。")
    else:
        res_df = st.session_state.current_res_df
        avg_strength = res_df["coal_uniaxial_compressive_strength_sigma_c"].mean()
        max_stress = res_df["coal_in_situ_stress_final_sigma"].max()
        max_k = res_df["stress_concentration_k"].max()

        c1, c2, c3 = st.columns(3)
        c1.metric("平均强度", f"{avg_strength:.2f} MPa")
        c2.metric("最大反演应力", f"{max_stress:.2f} MPa")
        c3.metric("最大 K 值", f"{max_k:.2f}")

        if st.button("生成技术日报"):
            report_query = "冲击地压 随钻参数 应力集中 卸压 效果检验 监测预警"
            db = get_rag_db()
            evidence = retrieve_evidence(db, report_query, k=4)
            prompt = build_report_prompt(res_df, evidence)

            st.subheader(f"冲击地压随钻反演与防冲诊断技术日报 ({datetime.now().strftime('%Y-%m-%d')})")
            placeholder = st.empty()
            stream_ollama(prompt, model_name, placeholder)
            render_evidence(evidence)
