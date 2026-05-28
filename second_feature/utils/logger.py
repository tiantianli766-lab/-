# -*- coding: utf-8 -*-
"""
矿井智能扩孔钻机随钻数据处理专用软件V1.0
日志管理模块：utils/logger.py
功能：满足软著全流程可追溯要求，实现四级日志分级管理、日志存储、查询与导出
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict
from datetime import datetime


class LoggerManager:
    """
    日志管理类：统一管理全数据处理流程的日志
    支持DEBUG、INFO、WARNING、ERROR四级日志，文件轮转，日志查询与导出
    """

    # 单例模式：确保全局只有一个日志管理器实例
    _instance: Optional['LoggerManager'] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_manager=None):
        """
        初始化日志管理器
        :param config_manager: 可选，配置管理器实例，不指定则使用默认配置
        """
        # 避免重复初始化
        if hasattr(self, 'initialized') and self.initialized:
            return

        self.config_manager = config_manager
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir: str = ""
        self._setup_log_dir()
        self.initialized = True

    def _setup_log_dir(self) -> None:
        """设置日志存储目录，不存在则创建"""
        if self.config_manager:
            base_dir = self.config_manager.get("storage.base_dir", "data")
            log_subdir = self.config_manager.get("storage.log_dir", "log")
            self.log_dir = os.path.join(base_dir, log_subdir)
        else:
            self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "log")
        os.makedirs(self.log_dir, exist_ok=True)

    def get_logger(self, module_name: str = "main") -> logging.Logger:
        """
        获取指定模块的日志记录器
        :param module_name: 模块名称，例如 "data_acquisition"、"signal_denoising"
        :return: 日志记录器实例
        """
        if module_name in self.loggers:
            return self.loggers[module_name]

        # 创建新的日志记录器
        logger = logging.getLogger(module_name)
        logger.setLevel(self._get_log_level())
        logger.propagate = False  # 避免日志重复输出

        # 清除已有的处理器，避免重复添加
        if logger.handlers:
            logger.handlers.clear()

        # 添加文件处理器：支持日志文件轮转
        file_handler = self._create_file_handler(module_name)
        logger.addHandler(file_handler)

        # 添加控制台处理器：仅在DEBUG级别输出到控制台
        console_handler = self._create_console_handler()
        logger.addHandler(console_handler)

        self.loggers[module_name] = logger
        return logger

    def _get_log_level(self) -> int:
        """
        获取配置的日志级别
        :return: logging日志级别常量
        """
        if self.config_manager:
            level_str = self.config_manager.get("log.level", "INFO")
        else:
            level_str = "INFO"

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        return level_map.get(level_str.upper(), logging.INFO)

    def _create_file_handler(self, module_name: str) -> RotatingFileHandler:
        """
        创建日志文件处理器，支持文件轮转
        :param module_name: 模块名称
        :return: RotatingFileHandler实例
        """
        log_file = os.path.join(self.log_dir, f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log")

        if self.config_manager:
            max_size = self.config_manager.get("log.max_file_size", 10 * 1024 * 1024)
            backup_count = self.config_manager.get("log.backup_count", 5)
            log_format = self.config_manager.get("log.format", "%(asctime)s - %(module)s - %(levelname)s - %(message)s")
        else:
            max_size = 10 * 1024 * 1024
            backup_count = 5
            log_format = "%(asctime)s - %(module)s - %(levelname)s - %(message)s"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self._get_log_level())
        return file_handler

    def _create_console_handler(self) -> logging.StreamHandler:
        """
        创建控制台日志处理器，仅在DEBUG级别输出
        :return: StreamHandler实例
        """
        console_handler = logging.StreamHandler()
        if self.config_manager:
            log_format = self.config_manager.get("log.format", "%(asctime)s - %(module)s - %(levelname)s - %(message)s")
        else:
            log_format = "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)  # 控制台仅输出DEBUG及以上级别
        return console_handler

    def log_data_processing_step(self, module_name: str, step_name: str, details: str = "") -> None:
        """
        记录数据处理步骤，满足软著可追溯要求
        :param module_name: 模块名称
        :param step_name: 处理步骤名称
        :param details: 可选，步骤详细信息
        """
        logger = self.get_logger(module_name)
        log_message = f"数据处理步骤: {step_name}"
        if details:
            log_message += f" | 详细信息: {details}"
        logger.info(log_message)

    def log_parameter_config(self, module_name: str, param_name: str, param_value: str) -> None:
        """
        记录参数配置信息
        :param module_name: 模块名称
        :param param_name: 参数名称
        :param param_value: 参数值
        """
        logger = self.get_logger(module_name)
        logger.debug(f"参数配置: {param_name} = {param_value}")

    def log_exception(self, module_name: str, exception: Exception, context: str = "") -> None:
        """
        记录异常信息，包含堆栈跟踪
        :param module_name: 模块名称
        :param exception: 异常实例
        :param context: 可选，异常发生上下文
        """
        logger = self.get_logger(module_name)
        log_message = f"异常发生: {str(exception)}"
        if context:
            log_message += f" | 上下文: {context}"
        logger.error(log_message, exc_info=True)

    def query_logs(self, module_name: str, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None, level: Optional[str] = None) -> List[str]:
        """
        查询指定模块的日志
        :param module_name: 模块名称
        :param start_time: 可选，开始时间
        :param end_time: 可选，结束时间
        :param level: 可选，日志级别
        :return: 符合条件的日志列表
        """
        log_file = os.path.join(self.log_dir, f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log")
        if not os.path.exists(log_file):
            return []

        logs = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # 简单的日志过滤（实际项目中可使用更复杂的解析）
                    if level and level.upper() not in line:
                        continue
                    if start_time or end_time:
                        try:
                            log_time_str = line.split(' - ')[0]
                            log_time = datetime.strptime(log_time_str, "%Y-%m-%d %H:%M:%S,%f")
                            if start_time and log_time < start_time:
                                continue
                            if end_time and log_time > end_time:
                                continue
                        except (ValueError, IndexError):
                            continue
                    logs.append(line)
        except Exception as e:
            self.log_exception("logger", e, "查询日志失败")
        return logs

    def export_logs(self, module_name: str, export_path: str,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    level: Optional[str] = None) -> bool:
        """
        导出日志到指定文件
        :param module_name: 模块名称
        :param export_path: 导出文件路径
        :param start_time: 可选，开始时间
        :param end_time: 可选，结束时间
        :param level: 可选，日志级别
        :return: 是否导出成功
        """
        logs = self.query_logs(module_name, start_time, end_time, level)
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(logs))
            return True
        except Exception as e:
            self.log_exception("logger", e, f"导出日志到 {export_path} 失败")
            return False