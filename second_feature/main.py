# -*- coding: utf-8 -*-
"""
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
入口模块：main.py
功能：集成导入、降噪、反演、导出全流程业务逻辑
"""
import pandas as pd
import os
from utils.config import ConfigManager
from utils.logger import LoggerManager
from core.data_acquisition import DataAcquisitionModule   # 新增导入
from core.data_validation import CoalDataValidator
from core.signal_denoising import SignalDenoisingModule
from core.coal_mechanics_inversion import CoalMechanicsInversion

class CoalDataAnalysisSystem:
    """随钻数据分析与力学反演系统：专注数据处理与参数计算"""
    def __init__(self):
        # 1. 初始化基础组件
        self.config_manager = ConfigManager()
        self.logger_manager = LoggerManager(self.config_manager)
        self.logger = self.logger_manager.get_logger("analysis_main")
        
        # 2. 初始化核心业务模块
        # 数据导入模块（内含：多源级联对齐算法）
        self.acquisition_module = DataAcquisitionModule(self.config_manager, self.logger_manager)
        
        # 数据校验模块
        self.validator = CoalDataValidator('drilling')
        
        # 降噪优化模块（内含：五层小波分解+自适应卡尔曼滤波）
        self.denoising_module = SignalDenoisingModule(self.config_manager, self.logger_manager)
        
        # 力学反演模块（内含：MSE模型计算+应力场迭代算法）
        self.mechanics_module = CoalMechanicsInversion(self.config_manager, self.logger_manager)
        
        # 3. 设置输出路径
        self.output_dir = self.config_manager.get("storage.report_out_dir", "./analysis_reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, input_file_path):
        """核心业务逻辑：文件导入对齐 -> 降噪 -> 反演 -> 报告生成"""
        task_id = f"ANALYSIS_{pd.Timestamp.now().strftime('%Y%n%d_%H%M%S')}"
        self.logger.info(f">>> 启动数据处理任务: {task_id}")

        try:
            # 第一阶段：数据导入与同步级联对齐
            # 使用 acquisition_module 而非 pd.read_csv，这能体现你的专利对齐算法
            self.logger.info("正在执行多源异构数据的时间戳同步级联对齐...")
            standard_data = self.acquisition_module.acquire_data(input_file_path)
            self.logger.info(f"数据载入并对齐成功，样本量: {len(standard_data)}")

            # 第二阶段：数据标准化校验
            self.logger.info("执行全局数据规范性校验与异常值清洗...")
            is_valid, cleaned_data = self.validator.validate_drilling_data(standard_data)
            
            # 第三阶段：多维信号协同抗干扰滤波 (核心算法)
            self.logger.info("执行db4小波五层分解与自适应卡尔曼协同降噪...")
            denoised_data = self.denoising_module.denoise_all_sensors(cleaned_data)

            # 第四阶段：煤体力学参数反演计算
            self.logger.info("基于MSE理论模型进行力学参数原地应力场反演...")
            results_df = self.mechanics_module.invert_all_points(denoised_data)

            # 第五阶段：结果持久化与生成导出分析报告
            output_filename = f"Inversion_Report_{task_id}.xlsx"
            output_path = os.path.join(self.output_dir, output_filename)
            results_df.to_excel(output_path, index=False)
            
            self.logger.info(f"任务圆满完成！分析报告已保存至: {output_path}")
            return results_df

        except Exception as e:
            self.logger.error(f"分析任务中途崩溃: {str(e)}", exc_info=True)
            return None

if __name__ == "__main__":
    # 模拟用户操作流程
    system = CoalDataAnalysisSystem()
    
    # 请确保在项目根目录下准备好 your_drilling_data.csv 
    # 或者修改成你的真实数据路径
    demo_file = "your_drilling_data.csv" 
    
    if os.path.exists(demo_file):
        system.process_file(demo_file)
    else:
        print(f"提示：未找到输入文件 {demo_file}，请将原始随钻数据放入根目录后再运行。")