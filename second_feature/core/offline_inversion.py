# -*- coding: utf-8 -*-
"""
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
软件简称：随钻数据处理软件
软件版本：V1.0
开发完成日期：2026年
模块名称：离线煤体力学参数批量反演模块
模块路径：core/offline_inversion.py
核心功能：离线监测数据批量反演、多算法优化、结果可视化、批量导出
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from typing import Tuple, List, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# 全局配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
EPS = 1e-6
DEFAULT_PARAM_BOUNDS = {
    'elastic_modulus': (1e3, 1e5),  # 弹性模量，MPa
    'poisson_ratio': (0.2, 0.4),  # 泊松比
    'cohesion': (1.0, 10.0),  # 内聚力，MPa
    'friction_angle': (20.0, 45.0),  # 内摩擦角，°
    'tensile_strength': (0.5, 5.0)  # 抗拉强度，MPa
}


class OfflineCoalMechanicsInversion:
    """离线煤体力学参数反演主类
    核心功能：
    1. 基于弹塑性力学理论的煤体正演计算
    2. 支持3种反演算法（差分进化/Nelder-Mead/最小二乘法）
    3. 多目标加权优化反演（应力+应变+位移）
    4. 反演结果可视化与误差分析
    5. 参数敏感性分析与批量反演
    """

    def __init__(self,
                 monitoring_data: pd.DataFrame,
                 geological_params: Dict[str, float],
                 param_bounds: Dict[str, Tuple[float, float]] = None,
                 inversion_method: str = 'differential_evolution',
                 weight_config: Dict[str, float] = None):
        """
        初始化反演模型
        :param monitoring_data: 现场监测数据 DataFrame
        :param geological_params: 地质基础参数
        :param param_bounds: 待反演参数范围
        :param inversion_method: 反演算法
        :param weight_config: 多目标优化权重配置
        """
        self.monitor_data = monitoring_data.copy()
        self.geo_params = geological_params
        self.param_bounds = param_bounds if param_bounds else DEFAULT_PARAM_BOUNDS
        self.inversion_method = inversion_method.lower()
        self.weight = weight_config if weight_config else {
            'strain': 0.4,
            'stress': 0.3,
            'displacement': 0.3
        }

        # 反演结果存储
        self.inverted_params = None
        self.inversion_history = []
        self.forward_calc_results = None
        self.error_metrics = None
        self.best_fitness = None

        # 数据预处理
        self._preprocess_monitor_data()
        self._check_input_validity()

    def _preprocess_monitor_data(self):
        """监测数据预处理：缺失值填充、异常值剔除"""
        required_cols = ['depth', 'stress_meas', 'strain_meas', 'displacement_meas']
        self.monitor_data = self.monitor_data.dropna(subset=required_cols)
        self.monitor_data = self.monitor_data[(self.monitor_data['stress_meas'] > 0) &
                                              (self.monitor_data['strain_meas'] > 0) &
                                              (self.monitor_data['displacement_meas'] > 0)]
        self.monitor_data.reset_index(drop=True, inplace=True)

    def _check_input_validity(self):
        """输入参数合法性校验"""
        required_geo = ['buried_depth', 'bulk_density', 'in_situ_stress', 'coal_thickness']
        for param in required_geo:
            if param not in self.geo_params:
                raise ValueError(f"缺失基础地质参数：{param}")
            if self.geo_params[param] <= 0:
                raise ValueError(f"地质参数{param}必须为正数")

        if abs(sum(self.weight.values()) - 1.0) > EPS:
            raise ValueError("多目标权重之和必须为1.0")

        valid_methods = ['differential_evolution', 'nelder_mead', 'least_squares']
        if self.inversion_method not in valid_methods:
            raise ValueError(f"不支持的反演算法，可选：{valid_methods}")

    def coal_forward_model(self, params: np.ndarray) -> np.ndarray:
        """
        煤体力学正演计算模型
        输入：[弹性模量, 泊松比, 内聚力, 内摩擦角, 抗拉强度]
        输出：计算应力、应变、位移
        """
        E, mu, c, phi, sigma_t = params
        H = self.geo_params['buried_depth']
        gamma = self.geo_params['bulk_density']
        sigma_0 = self.geo_params['in_situ_stress']
        h = self.geo_params['coal_thickness']

        depths = self.monitor_data['depth'].values
        n_samples = len(depths)
        calc_results = np.zeros((n_samples, 3))

        for i, z in enumerate(depths):
            # 应力计算（基于弹塑性理论）
            sigma_v = gamma * H * z / h
            sigma_h = mu / (1 - mu) * sigma_v
            sigma_calc = sigma_v + sigma_0

            # 应变计算
            epsilon_calc = sigma_calc / E + mu * sigma_h / E

            # 位移计算
            disp_calc = (sigma_calc * h) / (E * (1 - mu ** 2))

            calc_results[i] = [sigma_calc, epsilon_calc, disp_calc]

        return calc_results

    def objective_function(self, params: np.ndarray) -> float:
        """反演目标函数：多目标加权误差最小化"""
        calc_vals = self.coal_forward_model(params)
        meas_stress = self.monitor_data['stress_meas'].values
        meas_strain = self.monitor_data['strain_meas'].values
        meas_disp = self.monitor_data['displacement_meas'].values

        # 归一化误差计算
        stress_error = np.mean(np.abs((calc_vals[:, 0] - meas_stress) / (meas_stress + EPS)))
        strain_error = np.mean(np.abs((calc_vals[:, 1] - meas_strain) / (meas_strain + EPS)))
        disp_error = np.mean(np.abs((calc_vals[:, 2] - meas_disp) / (meas_disp + EPS)))

        # 加权总误差
        total_error = (self.weight['stress'] * stress_error +
                       self.weight['strain'] * strain_error +
                       self.weight['displacement'] * disp_error)

        return total_error

    def _inversion_de(self) -> np.ndarray:
        """差分进化算法反演（全局最优，推荐）"""
        bounds = list(self.param_bounds.values())
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=500,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        return result.x

    def _inversion_nelder(self) -> np.ndarray:
        """Nelder-Mead算法反演（局部优化）"""
        x0 = [np.mean(b) for b in self.param_bounds.values()]
        result = minimize(
            self.objective_function,
            x0=x0,
            method='Nelder-Mead',
            tol=1e-6,
            options={'maxiter': 1000}
        )
        return result.x

    def _inversion_ls(self) -> np.ndarray:
        """最小二乘法反演（线性近似）"""
        x0 = [np.mean(b) for b in self.param_bounds.values()]
        result = minimize(
            self.objective_function,
            x0=x0,
            method='L-BFGS-B',
            bounds=list(self.param_bounds.values()),
            tol=1e-6
        )
        return result.x

    def run_inversion(self) -> Dict[str, float]:
        """执行反演计算主流程"""
        print("=" * 60)
        print("开始离线煤体力学参数反演计算...")
        print(f"反演算法：{self.inversion_method}")
        print(f"监测样本数：{len(self.monitor_data)}")

        # 选择反演算法
        if self.inversion_method == 'differential_evolution':
            opt_params = self._inversion_de()
        elif self.inversion_method == 'nelder_mead':
            opt_params = self._inversion_nelder()
        elif self.inversion_method == 'least_squares':
            opt_params = self._inversion_ls()
        else:
            raise ValueError("无效的反演算法")

        # 存储反演结果
        param_names = list(self.param_bounds.keys())
        self.inverted_params = {k: round(v, 4) for k, v in zip(param_names, opt_params)}
        self.best_fitness = round(self.objective_function(opt_params), 6)
        self.forward_calc_results = self.coal_forward_model(opt_params)

        # 计算误差指标
        self._calculate_error_metrics()

        print("反演计算完成！")
        print(f"最优目标函数值：{self.best_fitness}")
        print(f"反演参数：{self.inverted_params}")
        print("=" * 60)

        return self.inverted_params

    def _calculate_error_metrics(self):
        """计算反演误差评价指标"""
        calc = self.forward_calc_results
        meas = self.monitor_data[['stress_meas', 'strain_meas', 'displacement_meas']].values

        mse = np.mean((calc - meas) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(calc - meas), axis=0)
        r2 = 1 - np.sum((calc - meas) ** 2, axis=0) / np.sum((meas - np.mean(meas, axis=0)) ** 2, axis=0)

        self.error_metrics = {
            'stress': {'MSE': mse[0], 'RMSE': rmse[0], 'MAE': mae[0], 'R2': r2[0]},
            'strain': {'MSE': mse[1], 'RMSE': rmse[1], 'MAE': mae[1], 'R2': r2[1]},
            'displacement': {'MSE': mse[2], 'RMSE': rmse[2], 'MAE': mae[2], 'R2': r2[2]}
        }

    def plot_inversion_comparison(self, save_path: str = None):
        """绘制监测值与计算值对比曲线"""
        if self.forward_calc_results is None:
            raise RuntimeError("请先执行反演计算run_inversion()")

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        depth = self.monitor_data['depth'].values
        meas = self.monitor_data[['stress_meas', 'strain_meas', 'displacement_meas']].values.T
        calc = self.forward_calc_results.T

        titles = ['应力对比', '应变对比', '位移对比']
        ylabels = ['应力(MPa)', '应变', '位移(m)']
        colors = ['#2E86AB', '#A23B72']

        for i, ax in enumerate(axes):
            ax.plot(depth, meas[i], 'o-', color=colors[0], label='现场监测', markersize=4)
            ax.plot(depth, calc[i], 's--', color=colors[1], label='反演计算', markersize=4)
            ax.set_title(titles[i], fontsize=12)
            ax.set_ylabel(ylabels[i], fontsize=10)
            ax.set_xlabel('埋藏深度(m)', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比曲线已保存至：{save_path}")
        plt.show()

    def plot_error_analysis(self, save_path: str = None):
        """误差分析可视化"""
        if self.error_metrics is None:
            raise RuntimeError("请先执行反演计算")

        metrics = ['RMSE', 'MAE', 'R2']
        stress_vals = [self.error_metrics['stress'][m] for m in metrics]
        strain_vals = [self.error_metrics['strain'][m] for m in metrics]
        disp_vals = [self.error_metrics['displacement'][m] for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.25

        ax.bar(x - width, stress_vals, width, label='应力', color='#2E86AB')
        ax.bar(x, strain_vals, width, label='应变', color='#A23B72')
        ax.bar(x + width, disp_vals, width, label='位移', color='#F18F01')

        ax.set_xlabel('误差指标', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title('煤体力学参数反演误差分析', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def export_results(self, save_dir: str = './inversion_results') -> str:
        """导出反演结果（JSON+Excel）"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_data = {
            'inversion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'inversion_method': self.inversion_method,
            'best_fitness': self.best_fitness,
            'inverted_parameters': self.inverted_params,
            'error_metrics': self.error_metrics,
            'sample_count': len(self.monitor_data),
            'geological_parameters': self.geo_params
        }

        # 导出JSON
        json_path = os.path.join(save_dir, f'inversion_result_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        # 导出Excel
        excel_path = os.path.join(save_dir, f'inversion_data_{timestamp}.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            self.monitor_data.to_excel(writer, sheet_name='监测数据', index=False)
            calc_df = pd.DataFrame(
                self.forward_calc_results,
                columns=['stress_calc', 'strain_calc', 'displacement_calc']
            )
            calc_df.to_excel(writer, sheet_name='计算结果', index=False)
            pd.DataFrame([self.inverted_params]).to_excel(writer, sheet_name='反演参数', index=False)

        print(f"反演结果已导出至：{save_dir}")
        return save_dir

    def get_parameter_sensitivity(self, perturb: float = 0.05) -> pd.DataFrame:
        """参数敏感性分析"""
        if self.inverted_params is None:
            raise RuntimeError("请先执行反演计算")

        base_params = np.array(list(self.inverted_params.values()))
        base_fitness = self.best_fitness
        param_names = list(self.inverted_params.keys())
        sensitivity = []

        for i in range(len(base_params)):
            # 正向扰动
            params_up = base_params.copy()
            params_up[i] *= (1 + perturb)
            fitness_up = self.objective_function(params_up)

            # 负向扰动
            params_down = base_params.copy()
            params_down[i] *= (1 - perturb)
            fitness_down = self.objective_function(params_down)

            # 敏感度计算
            sens = (abs(fitness_up - base_fitness) + abs(fitness_down - base_fitness)) / 2
            sensitivity.append([param_names[i], round(sens, 6)])

        sens_df = pd.DataFrame(sensitivity, columns=['parameter', 'sensitivity'])
        sens_df = sens_df.sort_values('sensitivity', ascending=False)
        return sens_df


def batch_inversion_process(monitor_files: List[str],
                            geo_params: Dict[str, float],
                            output_dir: str = './batch_results'):
    """批量反演处理流程"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_summary = []
    for file in monitor_files:
        print(f"\n正在处理文件：{file}")
        try:
            data = pd.read_csv(file) if file.endswith('.csv') else pd.read_excel(file)
            inversion = OfflineCoalMechanicsInversion(data, geo_params)
            params = inversion.run_inversion()
            inversion.export_results(output_dir)

            summary = {
                'file_name': os.path.basename(file),
                'inversion_success': True,
                'best_fitness': inversion.best_fitness,
                **params
            }
            results_summary.append(summary)
        except Exception as e:
            print(f"文件{file}处理失败：{str(e)}")
            results_summary.append({
                'file_name': os.path.basename(file),
                'inversion_success': False,
                'error': str(e)
            })

    # 导出批量结果汇总
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, 'batch_inversion_summary.xlsx')
    summary_df.to_excel(summary_path, index=False)
    print(f"\n批量反演汇总表已保存：{summary_path}")
    return summary_df


# 快速测试入口
if __name__ == '__main__':
    # 构造测试数据
    test_depth = np.linspace(50, 200, 20)
    test_data = pd.DataFrame({
        'depth': test_depth,
        'stress_meas': np.random.uniform(10, 30, 20),
        'strain_meas': np.random.uniform(1e-3, 5e-3, 20),
        'displacement_meas': np.random.uniform(0.01, 0.05, 20)
    })

    # 地质参数
    test_geo = {
        'buried_depth': 300.0,
        'bulk_density': 25.0,
        'in_situ_stress': 15.0,
        'coal_thickness': 6.0
    }

    # 执行反演
    inversion_model = OfflineCoalMechanicsInversion(test_data, test_geo)
    inversion_model.run_inversion()
    inversion_model.plot_inversion_comparison()
    inversion_model.plot_error_analysis()
    inversion_model.export_results()

    # 敏感性分析
    sens_result = inversion_model.get_parameter_sensitivity()
    print("\n参数敏感性分析结果：")
    print(sens_result)