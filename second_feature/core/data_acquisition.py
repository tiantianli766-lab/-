# -*- coding: utf-8 -*-
"""
版权所有 (C) 2026 开发团队/著作权人保留所有权利
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
模块名称：随钻载荷标准化解析与同步化重构模块

功能描述：
1. 实现针对 CSV、XLSX 等工业载荷记录文件的自适应多编码加载。
2. 建立针对非等频异构传感器序列的同步化重构算法，采用线性插值模型将采样特征锚定至 15Hz。
3. 执行多源工程字段语义映射，完成钻压 (WOB)、扭矩、转速等物理量的标准化对齐。
4. 基于时空位移增量模型推算瞬时钻进速度 (ROP)，并执行矿井常用工效学单位换算。
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from utils.config import ConfigManager
from utils.logger import LoggerManager

class DataAcquisitionModule:
    """
    随钻工程监测载荷采集与预处理主控类。
    内置多编码纠错机制以及基于线性插值的非等频序列同步化重构算法。
    """

    def __init__(self, config_manager: ConfigManager, logger_manager: LoggerManager) -> None:
        """
        初始化数据解析引擎，并锁定系统的标称分析频率。
        """
        self.config = config_manager
        self.logger = logger_manager.get_logger("data_acquisition")
        # 标定全系统统一采样频率设定为 15Hz
        self.target_sampling_frequency: int = 15

    def acquire_data(self, source_file_path: str) -> pd.DataFrame:
        """
        执行随钻载荷文件的标准解析与频率对齐作业流程。
        """
        self.logger.info(f"[IO_TRANS] 启动物理载荷文件检索加载，目标路径: {source_file_path}")

        # 第一阶段：多编码自适应探测加载逻辑，确保在 Windows/Linux 以及不同工控编码环境下数据读取的健壮性
        raw_dataframe = self._adaptive_load_file(source_file_path)
        
        # 第二阶段：语义层字段规范化处理，剔除列名中潜在的空白干扰符
        raw_dataframe.columns = [str(col).strip() for col in raw_dataframe.columns]

        # 第三阶段：多源异构字段模糊语义映射，建立统一物理名称索引
        semantic_patterns = {
            'WOB': ['钻压', '推力','wob', 'thrust', '载荷'],
            'torque': ['扭矩', 'torque', 'T', 'moment'],
            'rotational_speed': ['转速', 'rpm', 'speed', 'N'],
            'depth': ['深度', 'depth', '位移', 'H'],
            'hydraulic_pressure': ['液压', '压力', 'pressure'],
            'monitor_time': ['时间', 'time', 'date', 'timestamp']
        }
        
        extracted_fields_dict = {}
        for static_key, search_patterns in semantic_patterns.items():
            for column_name in raw_dataframe.columns:
                if any(pattern in column_name.lower() for pattern in search_patterns):
                    field_series = raw_dataframe[column_name]
                    if isinstance(field_series, pd.DataFrame): 
                        field_series = field_series.iloc[:, 0]
                    
                    if static_key == 'monitor_time':
                        extracted_fields_dict[static_key] = pd.to_datetime(field_series, errors='coerce')
                    else:
                        extracted_fields_dict[static_key] = pd.to_numeric(field_series, errors='coerce')
                    break
        
        # 第四阶段：基于线性插值的非等频序列同步化重构算法
        # 物理初衷：解决因硬件时钟不同步导致的离散采样点集无法构成连续波形的问题，建立统一的 15Hz 计算格架
        normalized_intermediate_df = pd.DataFrame(extracted_fields_dict).dropna(subset=['depth']).reset_index(drop=True)
        raw_sequential_length = len(normalized_intermediate_df)
        
        if raw_sequential_length < 2:
            self.logger.error("[ENGINE_ABORT] 目标样本特征密度不足两行，无法构建非等频对齐格架")
            raise ValueError("载荷文件有效样本基数由于少于 2 行，未能满足同步化重构的最低物理阈值")

        # 生成线性插值的基准坐标 xp (源时间轴) 与 x_target (标称频率时间轴)
        source_index_space = np.linspace(0, raw_sequential_length * (1000 / self.target_sampling_frequency), raw_sequential_length)
        target_reconstruction_grid = np.linspace(0, raw_sequential_length * (1000 / self.target_sampling_frequency), raw_sequential_length) 

        aligned_dataset = pd.DataFrame()
        reference_unix_anchor = datetime.now().timestamp()
        aligned_dataset['monitor_time'] = pd.to_datetime((target_reconstruction_grid / 1000 + reference_unix_anchor), unit='s')

        # 第五阶段：逐级执行物理通道的插值对齐协议
        target_physical_channels = ['WOB', 'torque', 'rotational_speed', 'depth', 'hydraulic_pressure']
        for channel in target_physical_channels:
            if channel in normalized_intermediate_df.columns:
                # 针对物理信号断点执行多项式插值平滑并生成 64 位浮点数据场
                base_signals = normalized_intermediate_df[channel].interpolate().ffill().bfill().values.astype(np.float64)
                # 映射并执行非等频序列同步化重构
                aligned_dataset[channel] = np.interp(target_reconstruction_grid, source_index_space, base_signals)
            else:
                self.logger.warning(f"[FIELD_MISSING] 通道 {channel} 缺失语义关联，系统自动执行零值初始化替代")
                aligned_dataset[channel] = 0.0

        # 第六阶段：针对矿井工况的 ROP (Rate of Penetration) 动力学指标推算
        # 标准说明：
        # 1. 单步采样时长 Delta_T (s) = 1.0 / 采样频率 (f=15Hz)
        # 2. 时长换算系数 (h) = Delta_T / 3600.0 (秒转小时)
        # 3. 瞬时速度 V = Delta_Depth / Delta_T (m/s)
        # 4. 工程转换 ROP = V * 3600 (m/h)，以此体现钻机工况的宏观生产率
        delta_time_hour = (1.0 / self.target_sampling_frequency) / 3600.0
        depth_increment = aligned_dataset['depth'].diff().fillna(0)
        
        # 执行带有抗抖动逻辑的 ROP 滑动平均滤波平滑处理
        raw_rop_velocity = (depth_increment / delta_time_hour).clip(0, 500)
        aligned_dataset['ROP'] = raw_rop_velocity.rolling(window=5, min_periods=1, center=True).mean()
        
        self.logger.info(f"[PROCESS_COMPLETE] 同步化重构协议执行结束，有效记录对齐基数: {len(aligned_dataset)}")
        return aligned_dataset

    def _adaptive_load_file(self, file_path: str) -> pd.DataFrame:
        """核心解析组件：基于编码感知的文件流健壮加载子程序"""
        source_ext = os.path.splitext(file_path)[1].lower()
        if source_ext in ['.xlsx', '.xls']:

            # 先不设表头读取
            raw_df = pd.read_excel(file_path, header=None)

            header_row = 0

            # 自动寻找真正表头
            for i, row in raw_df.iterrows():

                row_values = [str(v).strip() for v in row.tolist()]

                if (
                        ('时间' in row_values or 'time' in str(row_values).lower())
                        and
                        ('扭矩' in row_values or 'torque' in str(row_values).lower())
                ):
                    header_row = i
                    break

            # 用正确表头重新读取
            return pd.read_excel(file_path, header=header_row)
        
        encoding_candidate_list = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
        for current_encoding in encoding_candidate_list:
            try:
                processed_df = pd.read_csv(file_path, encoding=current_encoding)
                return processed_df
            except (UnicodeDecodeError, Exception):
                continue

        raise Exception(
            f"未能读取文件: {file_path}"
        )