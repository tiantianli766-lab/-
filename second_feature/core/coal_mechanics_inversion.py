# -*- coding: utf-8 -*-
"""
版权所有 (C) 2026 开发团队/著作权人保留所有权利
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
模块名称：煤体力学参数反向演化解算模块
功能描述：基于修正机械比能(MSE)模型与Mohr-Coulomb极限平衡理论，
          实现随钻载荷向原地应力、煤体强度及影响半径的非线性映射。
"""

import numpy as np
import pandas as pd
from utils.config import ConfigManager
from utils.logger import LoggerManager

class CoalMechanicsInversion:
    """
    针对随钻动能转化过程，反算地层本构特性与应力态的计算实体。
    采用修正的机械比能模型 (Modified MSE) 与解析法塑性展开圆柱模型。
    """

    def __init__(self, config_manager: ConfigManager, logger_manager: LoggerManager) -> None:
        """
        初始化反演算子。
        :param config_manager: 配置管理实例
        :param logger_manager: 日志管理实例
        """
        self.config = config_manager
        self.logger = logger_manager.get_logger("coal_mechanics_inversion")

        # 从配置文件预装载物理定数
        self.radius_base: float = self.config.get("coal_mechanics.initial_borehole_radius", 0.0625)
        self.internal_phi: float = np.deg2rad(self.config.get("coal_mechanics.friction_angle_default", 30.0))
        self.conversion_eff: float = self.config.get("coal_mechanics.mse_efficiency", 0.35)

    def invert_all_points(
        self, 
        input_data: pd.DataFrame, 
        h0: float = 500.0, 
        gamma: float = 24.5, 
        k_side: float = 1.2
    ) -> pd.DataFrame:
        """
        全量采样点力学参数张量反演解算协议。
        :param input_data: 经过净化处理的随钻数据帧
        :param h0: 测点参考基础深度 (m)
        :param gamma: 上覆岩层等效各向异性容重 (kN/m3)
        :param k_side: 侧向应力传递系数
        :return: 包含完整反演力学链的数据帧
        """
        self.logger.info(f"Execution logic initialized: ref_depth={h0}, lateral_coeff={k_side}")
        
        # 建立副本以保护原始序列，并强制备份关键原始载荷用于UI对比（_raw后缀）
        output_df = input_data.copy()
        for col in ['WOB', 'torque', 'rotational_speed', 'ROP']:
            if col in output_df.columns:
                # 如果采集模块尚未保留原始值，则将当前值视为原始参考（实际流程中应由前端或采集模块预留）
                if f"{col}_raw" not in output_df.columns:
                    output_df[f"{col}_raw"] = output_df[col].copy()
        
        # 1. 物理场空间参数初始化
        area_section = np.pi * (self.radius_base ** 2)
        # 规避分母零值风险
        safe_rop = output_df['ROP'].clip(lower=0.1)
        
        # 2. 机械比能 (Mechanical Specific Energy, MSE) 能量转换模型
        # 公式：MSE = F/A + (2*PI*N*T)/(A*ROP)
        # 压力分量 (MPa)
        thrust_component = output_df['WOB'] / (area_section * 1000.0)
        # 旋转扭矩分量 (MPa)，考虑单位换算 60(min->s) 与 1000(kN->MN)
        torque_component = (120 * np.pi * output_df['rotational_speed'] * output_df['torque']) \
                           / (area_section * safe_rop * 1000000.0)
        
        output_df['mse'] = thrust_component + torque_component

        # 3. 煤体单轴抗压强度理论反算 (Sigma_c)
        # 基于能量转化效率准则修正后的强度参数
        output_df['coal_uniaxial_compressive_strength_sigma_c'] = (
            output_df['mse'] * self.conversion_eff
        ).clip(lower=1.0, upper=100.0)

        # 4. 原地应力场 (In-situ Stress) 非线性重构
        # 基于地质埋深对原地垂直应力进行线性映射与修正
        vertical_pressure = (gamma * (h0 + output_df['depth']) * k_side) / 1000.0
        
        # 损伤演化变量 D 的引入（用于修正应力传递链）
        # 强度比值反映地层破碎劣化程度
        strength_ratio = output_df['coal_uniaxial_compressive_strength_sigma_c'] / output_df['mse'].clip(lower=0.1)
        damage_var = (1 - strength_ratio).clip(0, 1)
        # 传递因子修正：D越大，应力集中越明显
        xi_mod = 1.0 - (damage_var * 0.35)
        
        output_df['coal_in_situ_stress_final_sigma'] = vertical_pressure * xi_mod

        # 5. 塑性区半径理论推导 (基于 Mohr-Coulomb 极限平衡准则模型)
        # 计算系数 K = (1+sin phi) / (1-sin phi)
        k_const = (1 + np.sin(self.internal_phi)) / (1 - np.sin(self.internal_phi))
        
        sig_s = output_df['coal_in_situ_stress_final_sigma']
        sig_c = output_df['coal_uniaxial_compressive_strength_sigma_c']
        
        # 定义分级解算张量比率项
        # 公式参考：R_p = r_0 * [ (2*(Sigma_s*K + Sigma_c)) / (Sigma_c*(1+K)) ] ^ (1/(K-1))
        numerator = 2 * (sig_s * k_const + sig_c)
        denominator = sig_c * (1 + k_const)
        
        ratio_term = numerator / denominator.clip(lower=0.01)
        
        # 执行幂函数演化计算塑性扩张半径
        output_df['plastic_zone_radius_rp'] = self.radius_base * np.power(
            ratio_term.clip(lower=1.0001), 1.0 / (k_const - 1)
        )

        self.logger.info(f"Analytical inversion complete. Output records: {len(output_df)}")
        return output_df

    def calculate_energy_utilization(self, df: pd.DataFrame) -> float:
        """
        计算计算域内的全局能量利用率评价指标。
        """
        if 'mse' not in df.columns:
            return 0.0
        return float(df['mse'].mean() * self.conversion_eff)