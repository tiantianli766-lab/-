# -*- coding: utf-8 -*-
"""
软件全称：矿井智能扩孔钻机随钻数据处理专用软件
软件简称：随钻数据处理软件
软件版本：V1.0
开发完成日期：2026年
模块名称：数据存储与管理模块
模块路径：utils/data_storage.py
核心功能：多格式数据读写、SQLite数据库管理、数据备份恢复、版本控制
"""
import os
import json
import sqlite3
import hashlib
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 全局配置
DEFAULT_STORAGE_PATH = './data_storage'
DB_NAME = 'coal_mechanics_inversion.db'
SUPPORTED_FORMATS = ['csv', 'xlsx', 'json', 'parquet']
BACKUP_RETENTION_DAYS = 30
MAX_FILE_SIZE_MB = 100


class DataStorageError(Exception):
    """数据存储自定义异常"""
    pass


class DataStorageManager:
    """数据存储与管理主类"""

    def __init__(self, storage_path: str = None):
        """
        初始化存储管理器
        :param storage_path: 数据存储根目录
        """
        # 【修复：优先初始化所有属性，再执行初始化方法】
        self.operation_log = []
        self.storage_path = Path(storage_path) if storage_path else Path(DEFAULT_STORAGE_PATH)
        self.db_path = self.storage_path / DB_NAME

        # 执行初始化
        self._init_storage_structure()
        self._init_database()

    def _add_log(self, operation: str, status: str, details: str = ""):
        """添加操作日志"""
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation,
            'status': status,
            'details': details
        }
        self.operation_log.append(log_entry)

    def _init_storage_structure(self):
        """初始化存储目录结构"""
        dirs = [
            'monitoring_data',
            'inversion_results',
            'geological_params',
            'backups',
            'temp',
            'exports'
        ]
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            for dir_name in dirs:
                (self.storage_path / dir_name).mkdir(exist_ok=True)
            self._add_log('init_storage', 'success', f"存储目录初始化完成：{self.storage_path}")
        except Exception as e:
            raise DataStorageError(f"存储目录初始化失败：{str(e)}")

    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 监测数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    sample_count INTEGER,
                    data_range TEXT,
                    remark TEXT,
                    UNIQUE(file_name, data_hash)
                )
            ''')

            # 反演结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inversion_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_name TEXT NOT NULL,
                    create_time TEXT NOT NULL,
                    monitor_data_id INTEGER,
                    inversion_method TEXT,
                    best_fitness REAL,
                    params_json TEXT,
                    FOREIGN KEY (monitor_data_id) REFERENCES monitoring_data(id)
                )
            ''')

            # 地质参数表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS geological_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    param_name TEXT NOT NULL,
                    create_time TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    mine_area TEXT,
                    remark TEXT
                )
            ''')

            # 操作日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT
                )
            ''')

            conn.commit()
            conn.close()
            self._add_log('init_database', 'success', f"数据库初始化完成：{self.db_path}")
        except Exception as e:
            raise DataStorageError(f"数据库初始化失败：{str(e)}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值（MD5）"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _validate_file(self, file_path: Path) -> bool:
        """文件合法性校验"""
        if not file_path.exists():
            raise DataStorageError(f"文件不存在：{file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise DataStorageError(f"文件过大（{file_size_mb:.2f}MB），限制{MAX_FILE_SIZE_MB}MB")

        suffix = file_path.suffix.lower()[1:]
        if suffix not in SUPPORTED_FORMATS:
            raise DataStorageError(f"不支持的文件格式，可选：{SUPPORTED_FORMATS}")

        return True

    def save_monitoring_data(self,
                             data: pd.DataFrame,
                             file_name: str,
                             remark: str = "") -> int:
        """
        保存监测数据
        :return: 数据库记录ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_name = f"{Path(file_name).stem}_{timestamp}{Path(file_name).suffix}"
        save_path = self.storage_path / 'monitoring_data' / save_name

        try:
            # 保存文件
            suffix = Path(file_name).suffix.lower()[1:]
            if suffix == 'csv':
                data.to_csv(save_path, index=False, encoding='utf-8-sig')
            elif suffix == 'xlsx':
                data.to_excel(save_path, index=False)
            elif suffix == 'json':
                data.to_json(save_path, orient='records', force_ascii=False, indent=4)
            elif suffix == 'parquet':
                data.to_parquet(save_path, index=False)

            # 计算哈希并入库
            data_hash = self._calculate_file_hash(save_path)
            data_range = f"{data['depth'].min():.1f}-{data['depth'].max():.1f}m"

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO monitoring_data 
                (file_name, upload_time, data_hash, sample_count, data_range, remark)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (save_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  data_hash, len(data), data_range, remark))
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

            self._add_log('save_monitor', 'success', f"保存成功，记录ID：{record_id}")
            return record_id
        except Exception as e:
            self._add_log('save_monitor', 'error', str(e))
            raise DataStorageError(f"监测数据保存失败：{str(e)}")

    def load_monitoring_data(self, record_id: int = None, file_name: str = None) -> pd.DataFrame:
        """加载监测数据（按ID或文件名）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if record_id:
            cursor.execute('SELECT file_name FROM monitoring_data WHERE id = ?', (record_id,))
        elif file_name:
            cursor.execute('SELECT file_name FROM monitoring_data WHERE file_name = ?', (file_name,))
        else:
            raise DataStorageError("必须指定record_id或file_name")

        result = cursor.fetchone()
        conn.close()

        if not result:
            raise DataStorageError("未找到对应监测数据记录")

        file_path = self.storage_path / 'monitoring_data' / result[0]
        suffix = file_path.suffix.lower()[1:]

        if suffix == 'csv':
            return pd.read_csv(file_path)
        elif suffix == 'xlsx':
            return pd.read_excel(file_path)
        elif suffix == 'json':
            return pd.read_json(file_path, orient='records')
        elif suffix == 'parquet':
            return pd.read_parquet(file_path)

    def save_inversion_result(self,
                              result_name: str,
                              inverted_params: Dict[str, float],
                              monitor_data_id: int = None,
                              inversion_method: str = "",
                              best_fitness: float = None,
                              extra_data: Dict = None) -> int:
        """保存反演结果"""
        try:
            params_json = json.dumps(inverted_params, ensure_ascii=False, indent=4)
            if extra_data:
                params_json = json.dumps({**inverted_params, **extra_data}, ensure_ascii=False, indent=4)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO inversion_results 
                (result_name, create_time, monitor_data_id, inversion_method, best_fitness, params_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (result_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  monitor_data_id, inversion_method, best_fitness, params_json))
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

            # 同时保存JSON文件
            save_path = self.storage_path / 'inversion_results' / f"{result_name}_{record_id}.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(params_json)

            self._add_log('save_inversion', 'success', f"反演结果保存成功，ID：{record_id}")
            return record_id
        except Exception as e:
            self._add_log('save_inversion', 'error', str(e))
            raise DataStorageError(f"反演结果保存失败：{str(e)}")

    def save_geological_params(self,
                               param_name: str,
                               params: Dict[str, float],
                               mine_area: str = "",
                               remark: str = "") -> int:
        """保存地质参数"""
        try:
            params_json = json.dumps(params, ensure_ascii=False, indent=4)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO geological_params 
                (param_name, create_time, params_json, mine_area, remark)
                VALUES (?, ?, ?, ?, ?)
            ''', (param_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  params_json, mine_area, remark))
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

            self._add_log('save_geo', 'success', f"地质参数保存成功，ID：{record_id}")
            return record_id
        except Exception as e:
            self._add_log('save_geo', 'error', str(e))
            raise DataStorageError(f"地质参数保存失败：{str(e)}")

    def create_backup(self, backup_name: str = None) -> str:
        """创建数据备份"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = backup_name or f"backup_{timestamp}"
        backup_path = self.storage_path / 'backups' / backup_name

        try:
            # 备份数据库
            shutil.copy2(self.db_path, backup_path.parent / f"{backup_name}_db.db")
            # 备份关键数据目录
            for dir_name in ['monitoring_data', 'inversion_results', 'geological_params']:
                src_dir = self.storage_path / dir_name
                if src_dir.exists():
                    shutil.copytree(src_dir, backup_path / dir_name)

            # 清理过期备份
            self._cleanup_old_backups()

            self._add_log('backup', 'success', f"备份创建成功：{backup_name}")
            return str(backup_path)
        except Exception as e:
            self._add_log('backup', 'error', str(e))
            raise DataStorageError(f"备份创建失败：{str(e)}")

    def _cleanup_old_backups(self):
        """清理过期备份"""
        backup_dir = self.storage_path / 'backups'
        if not backup_dir.exists():
            return

        cutoff_time = datetime.now().timestamp() - (BACKUP_RETENTION_DAYS * 86400)
        for item in backup_dir.iterdir():
            if item.is_file() and item.stat().st_mtime < cutoff_time:
                item.unlink()
            elif item.is_dir() and item.stat().st_mtime < cutoff_time:
                shutil.rmtree(item)

    def export_data(self,
                    data_type: str,
                    record_id: int,
                    export_format: str = 'xlsx',
                    save_path: str = None) -> str:
        """数据导出"""
        if data_type == 'monitoring':
            data = self.load_monitoring_data(record_id=record_id)
        elif data_type == 'inversion':
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT params_json FROM inversion_results WHERE id = ?', (record_id,))
            result = cursor.fetchone()
            conn.close()
            if not result:
                raise DataStorageError("未找到反演结果")
            data = pd.DataFrame([json.loads(result[0])])
        else:
            raise DataStorageError(f"不支持的导出类型：{data_type}")

        save_path = Path(save_path) if save_path else self.storage_path / 'exports'
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = save_path / f"{data_type}_export_{record_id}_{timestamp}.{export_format}"

        if export_format == 'csv':
            data.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif export_format == 'xlsx':
            data.to_excel(file_path, index=False)
        elif export_format == 'json':
            data.to_json(file_path, orient='records', force_ascii=False, indent=4)

        self._add_log('export', 'success', f"数据导出成功：{file_path}")
        return str(file_path)

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM monitoring_data')
        monitor_count = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM inversion_results')
        inversion_count = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM geological_params')
        geo_count = cursor.fetchone()[0]

        conn.close()

        # 计算存储大小
        total_size = 0
        for item in self.storage_path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size

        return {
            'monitoring_data_count': monitor_count,
            'inversion_results_count': inversion_count,
            'geological_params_count': geo_count,
            'total_storage_size_mb': round(total_size / (1024 * 1024), 2),
            'backup_count': len(list((self.storage_path / 'backups').iterdir()))
        }


# 快速使用示例
if __name__ == '__main__':
    # 初始化存储管理器
    storage = DataStorageManager()

    # 1. 模拟保存监测数据
    test_data = pd.DataFrame({
        'depth': np.linspace(50, 200, 20),
        'stress_meas': np.random.uniform(10, 30, 20),
        'strain_meas': np.random.uniform(1e-3, 5e-3, 20),
        'displacement_meas': np.random.uniform(0.01, 0.05, 20)
    })
    monitor_id = storage.save_monitoring_data(test_data, 'test_monitor.csv', '测试数据')

    # 2. 模拟保存反演结果
    inversion_params = {
        'elastic_modulus': 25000.0,
        'poisson_ratio': 0.32,
        'cohesion': 5.5,
        'friction_angle': 35.0
    }
    inversion_id = storage.save_inversion_result(
        'test_inversion', inversion_params, monitor_id, 'differential_evolution', 0.0123
    )

    # 3. 查看存储统计
    stats = storage.get_storage_stats()
    print("存储统计：", stats)

    # 4. 创建备份
    backup_path = storage.create_backup()
    print("备份路径：", backup_path)