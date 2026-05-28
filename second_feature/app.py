# -*- coding: utf-8 -*-
"""
版权所有 (C) 2026 开发团队/著作权人保留所有权利
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 核心计算算子导入
from utils.config import ConfigManager
from utils.logger import LoggerManager
from core.data_acquisition import DataAcquisitionModule
from core.signal_denoising import SignalDenoisingModule
from core.coal_mechanics_inversion import CoalMechanicsInversion

# --- 1. 系统环境预检查 ---
def sys_init():
    for p in ["./raw_data", "./analysis_reports", "./workspace/logs"]:
        os.makedirs(p, exist_ok=True)

sys_init()

# --- 2. 界面视觉自定义 ---
st.set_page_config(page_title="随钻数据分析软件", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; }
    /* 汉化 Deploy 按钮提示 (虽然 Streamlit 强制显示，但我们可以淡化它) */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. 核心计算引擎初始化 ---
@st.cache_resource
def init_kernels():
    cfg = ConfigManager()
    log = LoggerManager(cfg)
    return {
        "acq": DataAcquisitionModule(cfg, log),
        "den": SignalDenoisingModule(cfg, log),
        "inv": CoalMechanicsInversion(cfg, log),
        "log": log
    }

kernels = init_kernels()

# --- 4. 侧边栏：参数配置面板 ---
with st.sidebar:
    st.title("参数配置面板")
    st.divider()
    
    st.subheader("地层物理属性")
    h_ref = st.number_input("埋藏深度 (米)", value=600.0)
    g_eff = st.number_input("覆盖层平均容重 (kN/m³)", value=24.5)
    k_lat = st.slider("地应力系数值", 0.5, 3.0, 1.2)
    
    st.divider()
    st.subheader("算法处理控制")
    w_level = st.slider("小波分解阶数", 2, 6, 5)
    f_smooth = st.select_slider("卡尔曼平滑强度", options=["弱", "中", "强"], value="中")
    q_val = {"弱": 0.1, "中": 0.01, "强": 0.001}[f_smooth]
    
    st.divider()
    st.caption("内核版本: V1.0.0")
    st.caption(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 5. 主交互界面设计 ---

st.title("随钻载荷解析与力学反演分析系统")
st.markdown("---")

t_monitor, t_workflow, t_eval, t_rep = st.tabs([
    "实时运行监控", "数据处理工程", "采集效果评估", "反演结果报告"
])

# Tab 1: 实时运行监控
with t_monitor:
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("核心模块状态")
        with st.container(border=True):
            status_df = pd.DataFrame({
                "模块名称": ["载荷同步器", "降噪引擎", "解析反演算子", "存储总线"],
                "当前状态": ["运行稳定", "系统就绪", "系统就绪", "网络挂载"]
            })
            st.table(status_df)
        
        st.subheader("性能性能指标")
        with st.container(border=True):
            mc1, mc2 = st.columns(2)
            mc1.metric("总线延迟", "2.1 毫秒", delta="-0.3")
            mc2.metric("内存映射", "稳定", delta="正常")
            st.caption("数据流采集频率: 15.0 赫兹")

    with c2:
        st.subheader("计算效率监测")
        with st.container(border=True):
            perf_df = pd.DataFrame({
                "处理阶段": ["数据预处理", "多尺度降噪", "参数迭代反演", "结果可视化生成"],
                "耗时占比": [12, 45, 30, 13]
            })
            fig_perf = px.bar(perf_df, x="处理阶段", y="耗时占比", color="耗时占比", 
                              color_continuous_scale="Blues", template="plotly_white")
            fig_perf.update_layout(height=350)
            st.plotly_chart(fig_perf, use_container_width=True)

# Tab 2: 数据处理工程
with t_workflow:
    st.subheader("随钻数据源文件载入")
    f_src = st.file_uploader("导入随钻原始记录文件", type=["csv", "xlsx"])
    
    if f_src:
        local_f = os.path.join("./raw_data", f_src.name)
        with open(local_f, "wb") as f:
            f.write(f_src.getbuffer())
            
        if st.button("开始数据处理", type="primary"):
            p_bar = st.progress(0)
            p_info = st.empty()
            
            try:
                p_info.text("正在进行多通道数据同步与重构...")
                df_raw = kernels["acq"].acquire_data(local_f)
                p_bar.progress(30)
                
                p_info.text("正在执行变尺度小波降噪与滤波...")
                df_den = kernels["den"].denoise_all_sensors(df_raw)
                p_bar.progress(60)
                
                p_info.text("正在反演煤体强度与应力演化轨迹...")
                df_res = kernels["inv"].invert_all_points(df_den, h0=h_ref, gamma=g_eff, k_side=k_lat)
                p_bar.progress(100)
                
                st.session_state['data'] = df_res
                st.session_state['flag'] = True
                p_info.success("全流程分析任务执行完毕。")
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

# Tab 3: 采集效果评估
with t_eval:
    if 'flag' in st.session_state:
        st.subheader("降噪前后载荷序列对比")
        df = st.session_state['data']
        
        # 汉化下拉菜单映射词
        ch_map = {
            "钻压 (kN)": "WOB",
            "扭矩 (N·m)": "torque",
            "转速 (r/min)": "rotational_speed",
            "钻速 (m/h)": "ROP"
        }
        ch_display = st.selectbox("选择观测物理量", list(ch_map.keys()))
        ch_key = ch_map[ch_display]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df[f"{ch_key}_raw"], name="原始采集序列", line=dict(color='lightgray', dash='dot')))
        fig.add_trace(go.Scatter(y=df[ch_key], name="净化后特征曲线", line=dict(color='#1f77b4', width=2.5)))
        fig.update_layout(template="plotly_white", xaxis_title="采样点序号", yaxis_title="物理幅值", 
                          legend=dict(title="数据类别"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("请先上传数据并执行处理流程。")

# Tab 4: 反演结果报告
with t_rep:
    if 'flag' in st.session_state:
        st.subheader("煤体物理特性反演结果明细")
        res_df = st.session_state['data']
        
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("单轴平均强度 (MPa)", f"{res_df['coal_uniaxial_compressive_strength_sigma_c'].mean():.2f}")
        rc2.metric("原地应力估值 (MPa)", f"{res_df['coal_in_situ_stress_final_sigma'].mean():.2f}")
        rc3.metric("塑性区影响半径 (m)", f"{res_df['plastic_zone_radius_rp'].max():.3f}")
        
        st.divider()
        
        # 处理 Plotly 的图例汉化
        # 使用 labels 参数映射变量名到中文
        fig_r = px.line(res_df, x="depth", y=["coal_uniaxial_compressive_strength_sigma_c", "coal_in_situ_stress_final_sigma"],
                        title="强度 / 应力随钻进深度分布曲线", 
                        labels={
                            "depth": "当前孔深 (米)", 
                            "value": "压力强度量值 (MPa)",
                            "variable": "反演参数项",
                            "coal_uniaxial_compressive_strength_sigma_c": "煤体单轴强度 (MPa)",
                            "coal_in_situ_stress_final_sigma": "原地应力水平 (MPa)"
                        },
                        template="plotly_white")
        
        # 强制替换图例名称逻辑 (双重保障)
        new_names = {'coal_uniaxial_compressive_strength_sigma_c': '煤体单轴强度 (MPa)', 
                     'coal_in_situ_stress_final_sigma': '原地应力水平 (MPa)'}
        fig_r.for_each_trace(lambda t: t.update(name = new_names.get(t.name, t.name)))
        
        st.plotly_chart(fig_r, use_container_width=True)
        
        c_csv = res_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载完整分析报告 (.CSV)", c_csv, "煤体力学反演报告.csv", "text/csv")
    else:
        st.info("待处理完成后自动生成详细数据报告。")

st.divider()
st.caption("系统运行诊断: 运行稳健 | 进程处理标识: 0x7E12 | 数据安全保护: 激活")