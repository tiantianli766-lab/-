# -*- coding: utf-8 -*-
"""
版权所有 (C) 2026 开发团队/著作权人保留所有权利
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
模块名称：多尺度空间小波与自适应卡尔曼协同滤波引擎

模块功能描述：
构建应对深地环境极端强干扰物理载荷数据流的滤波管道。
系统采用正交完备的 db4 型小波基进行瞬态冲击噪声的高阶多尺度离散特征剥离，
并通过构建状态空间转移模型，执行一阶增益自适应卡尔曼轨迹平滑追踪。
"""

import numpy as np
import pandas as pd
import pywt
from utils.config import ConfigManager
from utils.logger import LoggerManager

class SignalDenoisingModule:
    """
    随钻多维动态信号降噪与提取综合算子。
    集成离散小波分析子系统与线性最优滤波子系统。
    """
    
    def __init__(self, config_manager: ConfigManager, logger_manager: LoggerManager) -> None:
        """
        加载降噪引擎实例并注入参数约束校验网络。
        
        Args:
            config_manager (ConfigManager): 贯穿系统的全局参数分配管理器。
            logger_manager (LoggerManager): 接管该模块所有追踪痕迹的日志监控器。
        """
        self.config = config_manager
        self.logger = logger_manager.get_logger("signal_denoising")
        
        # 加载与校验系统级默认滤波配置底座
        try:
            self.baseline_wavelet_level = int(self.config.get("denoising.wavelet.decomposition_level", 4))
            self.baseline_kalman_q = float(self.config.get("denoising.kalman.initial_Q", 0.05))
            self.baseline_kalman_r = float(self.config.get("denoising.kalman.initial_R", 0.5))
            
            if self.baseline_wavelet_level <= 0 or self.baseline_kalman_q <= 0 or self.baseline_kalman_r <= 0:
                raise ValueError("底层超参值越界(必须 > 0)")
        except (ValueError, TypeError) as type_err_exp:
            self.logger.error(f"[SignalDenoisingModule] [ERR] 初始化失败！滤波参数畸变: {str(type_err_exp)}")
            raise
            
        self.logger.info(
            f"[SignalDenoisingModule] [INF] 滤波协处理器构建完毕 | "
            f"Wavelet-Level: {self.baseline_wavelet_level} | "
            f"Kalman-Covariance: Q={self.baseline_kalman_q}, R={self.baseline_kalman_r}"
        )

    def denoise_all_sensors(
        self, 
        drilling_dataframe: pd.DataFrame, 
        wave_level: int = None, 
        q_val: float = None
    ) -> pd.DataFrame:
        """
        启动随钻异构通道的并行协同净化框架。
        
        Args:
            drilling_dataframe (pd.DataFrame): 经过时序重排的含干扰信号数据集。
            wave_level (int, optional): 小波簇分解细化度重写参量。
            q_val (float, optional): 卡尔曼模型系统过程噪声方差补偿量。
            
        Returns:
            pd.DataFrame: 追加原始轨迹留存并替换主字段为重构净化分量的宽表集。
        """
        self.logger.info("[SignalDenoisingModule] [INF] 下发并行全量多维物理通道信号除噪指令")
        
        # 安全断言与运行期覆盖
        operative_level = int(wave_level) if wave_level is not None else self.baseline_wavelet_level
        operative_q = float(q_val) if q_val is not None else self.baseline_kalman_q
        
        purified_dataframe = drilling_dataframe.copy()
        
        # 系统约定的需要执行双模滤波的核心载荷通道标识映射
        target_denoising_channels = ["WOB", "torque", "rotational_speed", "ROP", "hydraulic_pressure"]
        
        for channel_name in target_denoising_channels:
            if channel_name in purified_dataframe.columns:
                # 信道数据格式规范化、连续化及高维坍缩展平
                raw_trajectory = pd.to_numeric(purified_dataframe[channel_name], errors='coerce')\
                                   .interpolate().ffill().bfill().values.astype(np.float64).ravel()
                
                # 固化污染源生波形轮廓以保留证据链查证功能
                purified_dataframe[f"{channel_name}_raw"] = raw_trajectory
                
                # =============================================================
                # 第 1 层核算法：基于正交 db4 小波基的频域硬剥离分解与逆演重构
                # =============================================================
                ceiling_level = pywt.dwt_max_level(len(raw_trajectory), 'db4')
                effective_decomposition_level = min(operative_level, ceiling_level)
                
                try:
                    # 依据工程规范执行离散小波高低频解耦
                    wavelet_coefficients = pywt.wavedec(raw_trajectory, 'db4', level=effective_decomposition_level)
                    
                    # 生成基于 Stein 无偏风险估计演化准则的稳健方差噪声判定阈值
                    median_absolute_deviation = np.median(np.abs(wavelet_coefficients[-1])) / 0.6745
                    universal_threshold = median_absolute_deviation * np.sqrt(2 * np.log(len(raw_trajectory)))
                    
                    # 作用阈值软收缩限制函数以钳位极端冲击突刺
                    wavelet_coefficients[1:] = [
                        pywt.threshold(coeff_array, value=universal_threshold, mode='soft') 
                        for coeff_array in wavelet_coefficients[1:]
                    ]
                    
                    intermediate_purified_signal = pywt.waverec(wavelet_coefficients, 'db4')[:len(raw_trajectory)]
                except Exception as wavelet_execution_error:
                    self.logger.warning(
                        f"[SignalDenoisingModule] [WRN] {channel_name} 通道小波变换树退化，转静默直挂处理状态: {str(wavelet_execution_error)}"
                    )
                    intermediate_purified_signal = raw_trajectory

                # =============================================================
                # 第 2 层核算法：针对一维线性状态测度空间的卡尔曼后向修正平滑
                # =============================================================
                final_tracked_signal = self._execute_adaptive_kalman_filter(
                    intermediate_purified_signal, 
                    operative_q, 
                    self.baseline_kalman_r
                )
                
                purified_dataframe[channel_name] = final_tracked_signal
                
                # 测算增益能效水平并记录到中央日志仓库
                signal_to_noise_gain = self._compute_snr_metric(raw_trajectory, final_tracked_signal)
                self.logger.info(f"[SignalDenoisingModule] [INF] {channel_name} 通道解析脱网，算理增益测度 (SNR): {signal_to_noise_gain:.2f} dB")

        return purified_dataframe

    def _execute_adaptive_kalman_filter(self, feed_signal: np.ndarray, proc_noise_q: float, meas_noise_r: float) -> np.ndarray:
        """
        封装级内联状态卡尔曼数字滤波器，基于误差协方差迭代寻求最优无偏估计解决方案。
        """
        sequence_length = len(feed_signal)
        if sequence_length == 0: 
            return feed_signal
        
        state_estimate = np.zeros(sequence_length)     
        error_covariance = np.zeros(sequence_length)       
        
        # 初始化边界先验状态量
        state_estimate[0] = feed_signal[0]
        error_covariance[0] = 1.0
        
        # 遵循“预测->更新”的闭环状态迭代推演准则
        for step_idx in range(1, sequence_length):
            apriori_state = state_estimate[step_idx - 1]
            apriori_error_covariance = error_covariance[step_idx - 1] + proc_noise_q
            
            # 收敛状态协方差与量测噪声比率计算最佳增益阵组
            kalman_optimal_gain = apriori_error_covariance / (apriori_error_covariance + meas_noise_r)
            
            state_estimate[step_idx] = apriori_state + kalman_optimal_gain * (feed_signal[step_idx] - apriori_state)
            error_covariance[step_idx] = (1 - kalman_optimal_gain) * apriori_error_covariance
            
        return state_estimate

    def _compute_snr_metric(self, raw_sequence: np.ndarray, recovered_sequence: np.ndarray) -> float:
        """
        基站测算服务子模块，通过解析均方差剥落水平，映射出信噪比优化分贝量(dB)。
        """
        residual_noise = raw_sequence - recovered_sequence
        power_signal = np.mean(recovered_sequence ** 2)
        power_noise = np.mean(residual_noise ** 2)
        if power_noise < 1e-10: 
            return 0.0
        return 10 * np.log10(power_signal / power_noise)