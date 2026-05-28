# -*- coding: utf-8 -*-
"""
版权所有 (C) 2026 开发团队/著作权人保留所有权利
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
包描述：系统核心算法组件库 (Core Algorithm Packages)

该初始化模块负责导出经过合规性重构后的核心计算模块，包括：
1. 载荷对齐与同步化重构模块 (Data Acquisition)
2. 信号多尺度净化与降噪模块 (Signal Denoising)
3. 煤体力学参数动态反演模块 (Mechanics Inversion)
4. 多维度数据质量校验模块 (Data Validation)
"""

# 从各子模块导入核心类
from .data_acquisition import DataAcquisitionModule
from .signal_denoising import SignalDenoisingModule
from .coal_mechanics_inversion import CoalMechanicsInversion
from .data_validation import CoalDataValidator

# 导出清单
__all__ = [
    "DataAcquisitionModule",
    "SignalDenoisingModule",
    "CoalMechanicsInversion",
    "CoalDataValidator"
]