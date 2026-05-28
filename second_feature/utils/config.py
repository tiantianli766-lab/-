# -*- coding: utf-8 -*-
"""
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
模块名称：全局配置管理模块
模块路径：utils/config.py
核心功能：
1. 离线文件读取与字段映射协议配置
2. 协同降噪算法（小波/卡尔曼）核心参数设定
3. 煤体力学反演定数与安全预警阈值管理
4. 满足软著“配置与逻辑分离”设计要求的参数管理
"""

import os
import json
from typing import Dict, Any, Optional

class ConfigManager:
    """
    系统配置管理类：统一管理分析软件的输入协议、算法参数及存储路径
    """
    def __init__(self, config_file_path: Optional[str] = None):
        # 默认配置文件存储路径
        self.default_config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        self.default_config_file = os.path.join(self.default_config_dir, "analysis_settings.json")

        # 初始化分析软件所需的默认配置
        self.config: Dict[str, Any] = self._init_default_config()

        # 加载逻辑：优先加载外部JSON，不存在则持久化默认配置
        if config_file_path and os.path.exists(config_file_path):
            self.load_config(config_file_path)
        elif os.path.exists(self.default_config_file):
            self.load_config(self.default_config_file)
        else:
            self._ensure_config_dir()
            self.save_config(self.default_config_file)

    def _init_default_config(self) -> Dict[str, Any]:
        """
        核心配置定义：完全脱离硬件，聚焦数据解析与计算规律
        """
        return {
            # ==================== 1. 数据解析要素配置 (Data Parsing) ====================
            "acquisition": {
                "sampling_frequency": 15,          # 处理频率：对齐至15Hz标准时间轴
                "input_format": "csv/xlsx",        # 支持的离线文件格式
                "auto_alignment": True,            # 是否开启多源数据时间戳同步级联对齐
                "sensors": [                       # 这里的Sensor指代数据文件中的“特征维度”
                    {"name": "钻压传感器", "unit": "kN", "range_min": 0, "range_max": 200, "mapping_key": "WOB"},
                    {"name": "扭矩传感器", "unit": "N·m", "range_min": 0, "range_max": 5000, "mapping_key": "Torque"},
                    {"name": "转速传感器", "unit": "r/min", "range_min": 0, "range_max": 300, "mapping_key": "RPM"},
                    {"name": "位移传感器", "unit": "m", "range_min": 0, "range_max": 100, "mapping_key": "Depth"},
                    {"name": "液压压力传感器", "unit": "MPa", "range_min": 0, "range_max": 50, "mapping_key": "Pressure"}
                ]
            },

            # ==================== 2. 信号处理算法配置 (Signal Processing) ====================
            "denoising": {
                "wavelet": {
                    "wavelet_name": "db4",        # 多尺度分析所采用的小波基
                    "decomposition_level": 5,     # 五层细节系数分解（专利要求的核心参数）
                    "threshold_method": "soft"    # 软阈值收敛算法，用于滤除高频突刺
                },
                "kalman": {
                    "initial_Q": 0.005,           # 过程噪声协方差：控制系统灵敏度
                    "initial_R": 0.08,            # 测量噪声协方差：控制滤波平滑度
                    "adaptive_mode": True         # 开启自适应残差校正模式
                }
            },

            # ==================== 3. 煤体力学反演参数 (Inversion Constants) ====================
            "coal_mechanics": {
                "initial_borehole_radius": 0.0625, # 初始孔径基准（单位：m）
                "friction_angle_default": 32.0,    # 煤体内摩擦角默认估值
                "confining_pressure_ref": 4.5,     # 地质有效围压参考值（MPa）
                "mse_efficiency": 0.35,            # 机械比能(MSE)转化效率系数
                "warning_thresholds": {
                    "stress_alert": 0.75,          # 原地应力预警比例（达到阈值的75%）
                    "plastic_ratio": 1.25          # 塑性区半径扩张倍数预警值
                }
            },

            # ==================== 4. 存储与IO路径配置 (Storage & IO) ====================
            "storage": {
                "base_dir": "./data_analysis_workspace",
                "input_raw_dir": "raw_data",       # 待处理原始数据存放处
                "report_out_dir": "analysis_reports", # 分析报告输出路径
                "db_cache": "analysis_history.db"  # 分析历史SQLite数据库
            },

            # ==================== 5. 运行日志设置 (Logging) ====================
            "log": {
                "level": "INFO",
                "format": "%(asctime)s - [%(levelname)s] - %(message)s"
            }
        }

    def _ensure_config_dir(self) -> None:
        """创建配置目录"""
        if not os.path.exists(self.default_config_dir):
            os.makedirs(self.default_config_dir, exist_ok=True)

    def load_config(self, file_path: str) -> None:
        """加载外部JSON配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.config.update(json.load(f))
        except Exception:
            pass # 加载失败则保持默认配置

    def save_config(self, file_path: str) -> None:
        """保存当前配置到JSON"""
        try:
            self._ensure_config_dir()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"配置保存失败: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        字段检索：支持 "denoising.wavelet.decomposition_level" 形式访问
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
