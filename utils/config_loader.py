import os
import copy
import yaml

DEFAULT_CONFIG = {
    "modules": {
        "deduplication": {
            "enabled": True,
            "phash_hash_size": 16,
            "phash_threshold": 10,
            "ssim_threshold": 0.95,
            "ssim_win_size": 7,
            "use_phash_prefilter": True,
        },
        "deblur": {
            "enabled": True,
            "bren_threshold": 50,
            "resize_for_analysis": [512, 512],
        },
        "anomaly_detection": {
            "enabled": True,
            "overexposed_threshold": 240,
            "underexposed_threshold": 15,
            "all_black_threshold": 5,
            "all_white_threshold": 250,
            "low_entropy_threshold": 2,
            "low_saturation_threshold": 10,
            "uniform_std_threshold": 5,
        },
    },
    "concurrency": {
        "enabled": True,
        "num_workers": 0,
        "batch_size": 100,
        "memory_limit_mb": 2048,
    },
    "progress": {
        "enabled": True,
        "update_interval": 10,
    },
    "file": {
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"],
        "recursive": False,
        "output_dir": "output",
        "csv_encoding": "utf-8-sig",
    },
    "logging": {
        "level": "INFO",
        "max_bytes": 10485760,
        "backup_count": 3,
        "format": "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 中的值覆盖 base，base 中不存在的键保留默认"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _validate(cfg: dict) -> None:
    """校验关键参数合法性，不合法抛 ValueError"""
    dedup = cfg["modules"]["deduplication"]
    if dedup["ssim_win_size"] % 2 == 0:
        raise ValueError(f"ssim_win_size 必须为奇数，当前值：{dedup['ssim_win_size']}")
    if cfg["concurrency"]["num_workers"] < 0:
        raise ValueError(f"num_workers 不能为负数，当前值：{cfg['concurrency']['num_workers']}")
    if cfg["concurrency"]["batch_size"] < 1:
        raise ValueError(f"batch_size 必须 >= 1，当前值：{cfg['concurrency']['batch_size']}")


def load_config(config_path: str | None) -> dict:
    """
    加载配置文件，与默认配置深度合并后验证。
    config_path 为 None 或文件不存在时，返回默认配置。
    """
    user_cfg = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    _validate(cfg)
    return cfg
