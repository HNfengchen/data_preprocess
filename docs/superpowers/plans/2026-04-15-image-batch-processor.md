# 图片批量处理工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建图片批量处理工具，支持去重（pHash+SSIM）、去模糊（BREN）、异常检测（熵+亮度+HSV）三个可选模块，并行处理，输出筛选图片和CSV分析报告。

**Architecture:** 插件式模块架构，每个处理模块继承BaseModule接口。主流程通过ProcessPoolExecutor并行处理各文件夹内图片，批次处理控制内存。配置驱动，YAML文件控制所有模块参数。

**Tech Stack:** Python 3.8+, OpenCV, Pillow, scikit-image, imagehash, NumPy, PyYAML, tqdm, psutil, concurrent.futures

---

## 文件结构

```
data_preprocess/
├── config.yaml                     # 主配置文件
├── main.py                         # CLI入口，串联所有模块
├── requirements.txt                # 依赖声明
├── modules/
│   ├── __init__.py                 # 导出所有模块类
│   ├── base.py                     # BaseModule抽象基类
│   ├── deduplication.py            # pHash预筛+SSIM精比对去重
│   ├── deblur.py                   # BREN锐度检测去模糊
│   └── anomaly_detection.py        # 熵+亮度+HSV异常检测
├── utils/
│   ├── __init__.py
│   ├── config_loader.py            # YAML加载+验证+默认值合并
│   ├── logger.py                   # 轮转日志，同时输出控制台和文件
│   ├── memory_monitor.py           # psutil内存监控+GC触发
│   ├── progress.py                 # tqdm进度条封装
│   └── file_handler.py             # 图片扫描+验证+输出目录创建+CSV写入
└── tests/
    ├── __init__.py
    ├── conftest.py                 # pytest fixtures（测试图片生成）
    ├── test_config_loader.py
    ├── test_file_handler.py
    ├── test_deduplication.py
    ├── test_deblur.py
    └── test_anomaly_detection.py
```

---

## Task 1: 项目初始化 + requirements.txt

**Files:**
- Create: `requirements.txt`
- Create: `modules/__init__.py`
- Create: `utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 初始化 git 仓库**

```bash
git init
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo "output/" >> .gitignore
echo "logs/" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo "*.egg-info/" >> .gitignore
```

- [ ] **Step 2: 创建 requirements.txt**

```
opencv-python>=4.5.0
numpy>=1.20.0
Pillow>=8.0.0
scikit-image>=0.18.0
scipy>=1.7.0
PyYAML>=5.4.0
tqdm>=4.60.0
psutil>=5.8.0
imagehash>=4.2.0
pytest>=7.0.0
```

- [ ] **Step 3: 安装依赖**

```bash
pip install -r requirements.txt
```

期望：无报错，所有包安装成功。

- [ ] **Step 4: 创建空 __init__.py 文件**

`modules/__init__.py`:
```python
from .deduplication import DeduplicationModule
from .deblur import DeblurModule
from .anomaly_detection import AnomalyDetectionModule

__all__ = ["DeduplicationModule", "DeblurModule", "AnomalyDetectionModule"]
```

`utils/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: 创建 tests/conftest.py（测试 fixtures）**

```python
import pytest
import numpy as np
import cv2
import os
import tempfile


@pytest.fixture
def tmp_image_dir():
    """创建临时图片目录，测试后自动清理"""
    with tempfile.TemporaryDirectory() as d:
        yield d


def make_image(path: str, width: int = 100, height: int = 100,
               color: tuple = (128, 128, 128), blur_sigma: float = 0.0):
    """生成测试用图片文件"""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    if blur_sigma > 0:
        ksize = int(blur_sigma * 6) | 1  # 保证奇数
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def sample_sharp_image(tmp_image_dir):
    """清晰图片（高频纹理）"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[::2, ::2] = 255  # 棋盘格纹理，高锐度
    path = os.path.join(tmp_image_dir, "sharp.jpg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def sample_blurry_image(tmp_image_dir):
    """模糊图片（纯色+大高斯模糊）"""
    path = os.path.join(tmp_image_dir, "blurry.jpg")
    return make_image(path, color=(100, 100, 100), blur_sigma=15.0)


@pytest.fixture
def sample_normal_image(tmp_image_dir):
    """正常图片（中等亮度，有纹理）"""
    img = np.random.randint(80, 180, (100, 100, 3), dtype=np.uint8)
    path = os.path.join(tmp_image_dir, "normal.jpg")
    cv2.imwrite(path, img)
    return path
```

- [ ] **Step 6: 验证 pytest 可运行**

```bash
pytest tests/ -v
```

期望：`no tests ran` 或 `0 passed`，无错误。

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "chore: project scaffold with requirements and test fixtures"
```

---

## Task 2: utils/config_loader.py

**Files:**
- Create: `utils/config_loader.py`
- Create: `tests/test_config_loader.py`

- [ ] **Step 1: 写失败测试**

`tests/test_config_loader.py`:
```python
import pytest
import os
import yaml
import tempfile
from utils.config_loader import load_config, DEFAULT_CONFIG


def write_yaml(d: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(d, f)


class TestLoadConfig:
    def test_returns_default_when_file_missing(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg["modules"]["deduplication"]["enabled"] is True
        assert cfg["concurrency"]["batch_size"] == 100

    def test_merges_user_values_over_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        write_yaml({"modules": {"deblur": {"bren_threshold": 99}}}, str(p))
        cfg = load_config(str(p))
        assert cfg["modules"]["deblur"]["bren_threshold"] == 99
        # 其他默认值保留
        assert cfg["modules"]["deduplication"]["enabled"] is True

    def test_raises_on_invalid_ssim_win_size(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        write_yaml({"modules": {"deduplication": {"ssim_win_size": 4}}}, str(p))
        with pytest.raises(ValueError, match="ssim_win_size"):
            load_config(str(p))

    def test_raises_on_negative_workers(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        write_yaml({"concurrency": {"num_workers": -1}}, str(p))
        with pytest.raises(ValueError, match="num_workers"):
            load_config(str(p))

    def test_default_config_is_valid(self):
        """DEFAULT_CONFIG 自身通过验证"""
        import copy
        cfg = load_config.__wrapped__(copy.deepcopy(DEFAULT_CONFIG)) \
            if hasattr(load_config, "__wrapped__") else load_config(None)
        assert cfg is not None
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_config_loader.py -v
```

期望：`ImportError: No module named 'utils.config_loader'`

- [ ] **Step 3: 实现 utils/config_loader.py**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_config_loader.py -v
```

期望：5 passed。

- [ ] **Step 5: Commit**

```bash
git add utils/config_loader.py tests/test_config_loader.py
git commit -m "feat: config loader with deep merge and validation"
```

---

## Task 3: utils/logger.py

**Files:**
- Create: `utils/logger.py`

（logger 功能简单，与外部无接口依赖，直接实现，不需额外单测。）

- [ ] **Step 1: 实现 utils/logger.py**

```python
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_dir: str = "logs", config: dict | None = None) -> logging.Logger:
    """
    创建或获取具名 logger。
    同时输出到控制台（INFO+）和轮转文件（DEBUG+）。
    """
    if config is None:
        config = {}

    level_str = config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    max_bytes = config.get("max_bytes", 10 * 1024 * 1024)
    backup_count = config.get("backup_count", 3)
    fmt = config.get("format", "[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
    datefmt = config.get("date_format", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 已初始化，直接返回

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 handler（轮转）
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
```

- [ ] **Step 2: 手动验证 logger 不崩溃**

```bash
python -c "from utils.logger import setup_logger; l=setup_logger('test'); l.info('ok')"
```

期望：控制台打印 `[...][INFO][test] ok`，生成 `logs/app.log`。

- [ ] **Step 3: Commit**

```bash
git add utils/logger.py
git commit -m "feat: rotating logger with console+file output"
```

---

## Task 4: utils/memory_monitor.py

**Files:**
- Create: `utils/memory_monitor.py`

- [ ] **Step 1: 实现 utils/memory_monitor.py**

```python
import gc
import psutil


class MemoryMonitor:
    """监控当前进程内存，超限时触发 GC"""

    def __init__(self, limit_mb: int = 2048):
        self.limit_mb = limit_mb

    def get_usage_mb(self) -> float:
        """返回当前进程 RSS（MB）"""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def check_and_collect(self) -> bool:
        """
        内存超限则触发 GC。
        返回 True 表示触发了 GC，False 表示未超限。
        """
        if self.get_usage_mb() > self.limit_mb:
            gc.collect()
            return True
        return False
```

- [ ] **Step 2: 验证可导入**

```bash
python -c "from utils.memory_monitor import MemoryMonitor; m=MemoryMonitor(); print(m.get_usage_mb())"
```

期望：输出一个正数（当前内存 MB）。

- [ ] **Step 3: Commit**

```bash
git add utils/memory_monitor.py
git commit -m "feat: memory monitor with GC trigger"
```

---

## Task 5: utils/progress.py

**Files:**
- Create: `utils/progress.py`

- [ ] **Step 1: 实现 utils/progress.py**

```python
from tqdm import tqdm


class ProgressTracker:
    """tqdm 进度条封装，支持禁用模式（测试/CI）"""

    def __init__(self, total: int, desc: str = "处理中", enabled: bool = True):
        self._enabled = enabled
        self.pbar = tqdm(total=total, desc=desc, unit="张", disable=not enabled)

    def update(self, n: int = 1, msg: str = "") -> None:
        self.pbar.update(n)
        if msg:
            self.pbar.set_postfix_str(msg)

    def close(self) -> None:
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

- [ ] **Step 2: 验证可导入**

```bash
python -c "from utils.progress import ProgressTracker; p=ProgressTracker(3,'test',False); [p.update() for _ in range(3)]; p.close(); print('ok')"
```

期望：打印 `ok`，无崩溃。

- [ ] **Step 3: Commit**

```bash
git add utils/progress.py
git commit -m "feat: tqdm progress tracker wrapper"
```

---

## Task 6: utils/file_handler.py

**Files:**
- Create: `utils/file_handler.py`
- Create: `tests/test_file_handler.py`

- [ ] **Step 1: 写失败测试**

`tests/test_file_handler.py`:
```python
import os
import csv
import pytest
from tests.conftest import make_image
from utils.file_handler import scan_images, validate_image, create_output_dirs, write_csv


class TestScanImages:
    def test_finds_jpg_and_png(self, tmp_image_dir):
        make_image(os.path.join(tmp_image_dir, "a.jpg"))
        make_image(os.path.join(tmp_image_dir, "b.png"))
        open(os.path.join(tmp_image_dir, "c.txt"), "w").close()
        result = scan_images(tmp_image_dir, [".jpg", ".png"])
        basenames = {os.path.basename(p) for p in result}
        assert basenames == {"a.jpg", "b.png"}

    def test_recursive_false_skips_subdirs(self, tmp_image_dir):
        sub = os.path.join(tmp_image_dir, "sub")
        os.makedirs(sub)
        make_image(os.path.join(sub, "nested.jpg"))
        make_image(os.path.join(tmp_image_dir, "root.jpg"))
        result = scan_images(tmp_image_dir, [".jpg"], recursive=False)
        basenames = {os.path.basename(p) for p in result}
        assert basenames == {"root.jpg"}

    def test_recursive_true_includes_subdirs(self, tmp_image_dir):
        sub = os.path.join(tmp_image_dir, "sub")
        os.makedirs(sub)
        make_image(os.path.join(sub, "nested.jpg"))
        make_image(os.path.join(tmp_image_dir, "root.jpg"))
        result = scan_images(tmp_image_dir, [".jpg"], recursive=True)
        assert len(result) == 2

    def test_empty_dir_returns_empty(self, tmp_image_dir):
        assert scan_images(tmp_image_dir, [".jpg"]) == []


class TestValidateImage:
    def test_valid_image_returns_true(self, sample_normal_image):
        valid, err = validate_image(sample_normal_image)
        assert valid is True
        assert err == ""

    def test_corrupt_file_returns_false(self, tmp_image_dir):
        p = os.path.join(tmp_image_dir, "bad.jpg")
        with open(p, "wb") as f:
            f.write(b"not an image at all")
        valid, err = validate_image(p)
        assert valid is False
        assert err != ""


class TestWriteCsv:
    def test_writes_rows_with_header(self, tmp_image_dir):
        out = os.path.join(tmp_image_dir, "report.csv")
        rows = [{"filename": "a.jpg", "final_keep": "是"}, {"filename": "b.jpg", "final_keep": "否"}]
        write_csv(rows, out, encoding="utf-8-sig")
        with open(out, encoding="utf-8-sig") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["filename"] == "a.jpg"
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_file_handler.py -v
```

期望：`ImportError`

- [ ] **Step 3: 实现 utils/file_handler.py**

```python
import os
import csv
from PIL import Image
from typing import List, Tuple


def scan_images(folder: str, extensions: List[str], recursive: bool = False) -> List[str]:
    """扫描文件夹，返回所有匹配扩展名的图片路径列表（排序）"""
    exts = {e.lower() for e in extensions}
    result = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    result.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in exts:
                full = os.path.join(folder, f)
                if os.path.isfile(full):
                    result.append(full)
    return sorted(result)


def validate_image(file_path: str) -> Tuple[bool, str]:
    """
    验证图片是否可读且完整。
    返回 (True, "") 或 (False, 错误信息)。
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
        return True, ""
    except Exception as e:
        return False, str(e)


def create_output_dirs(base_output: str, folder_name: str) -> Tuple[str, str]:
    """
    在 base_output/folder_name/ 下创建 filtered_images/ 目录。
    返回 (filtered_images路径, csv路径)。
    """
    filtered_dir = os.path.join(base_output, folder_name, "filtered_images")
    os.makedirs(filtered_dir, exist_ok=True)
    csv_path = os.path.join(base_output, folder_name, "analysis_report.csv")
    return filtered_dir, csv_path


def write_csv(rows: List[dict], output_path: str, encoding: str = "utf-8-sig") -> None:
    """将 rows（字典列表）写入 CSV，自动从第一行推断列名"""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_file_handler.py -v
```

期望：所有测试通过。

- [ ] **Step 5: Commit**

```bash
git add utils/file_handler.py tests/test_file_handler.py
git commit -m "feat: file handler - scan, validate, output dirs, csv writer"
```

---

## Task 7: modules/base.py

**Files:**
- Create: `modules/base.py`

- [ ] **Step 1: 实现 modules/base.py**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModule(ABC):
    """所有处理模块的抽象基类，定义统一接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled: bool = config.get("enabled", True)

    @abstractmethod
    def process(self, image_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单张图片，更新并返回 record 字典。
        record 已包含基础字段（filename, file_path 等）。
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """模块名称，用于日志"""
        raise NotImplementedError
```

- [ ] **Step 2: 验证可导入**

```bash
python -c "from modules.base import BaseModule; print('ok')"
```

期望：输出 `ok`。

- [ ] **Step 3: Commit**

```bash
git add modules/base.py
git commit -m "feat: BaseModule abstract interface"
```

---

## Task 8: modules/deblur.py

**Files:**
- Create: `modules/deblur.py`
- Create: `tests/test_deblur.py`

- [ ] **Step 1: 写失败测试**

`tests/test_deblur.py`:
```python
import pytest
import numpy as np
import cv2
import os
from modules.deblur import DeblurModule


DEFAULT_CFG = {
    "enabled": True,
    "bren_threshold": 50,
    "resize_for_analysis": [512, 512],
}


def make_sharp_image(path):
    """棋盘格 = 高梯度 = 高锐度"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[::4, :] = 255
    img[:, ::4] = 255
    cv2.imwrite(path, img)
    return path


def make_blurry_image(path):
    """纯色大模糊 = 低梯度 = 低锐度"""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img = cv2.GaussianBlur(img, (51, 51), 20)
    cv2.imwrite(path, img)
    return path


class TestDeblurModule:
    def test_sharp_image_not_blurry(self, tmp_image_dir):
        path = make_sharp_image(os.path.join(tmp_image_dir, "sharp.jpg"))
        mod = DeblurModule(DEFAULT_CFG)
        rec = {"file_path": path}
        result = mod.process(path, rec)
        assert result["is_blurry"] == "否"
        assert result["bren_sharpness"] > DEFAULT_CFG["bren_threshold"]

    def test_blurry_image_detected(self, tmp_image_dir):
        path = make_blurry_image(os.path.join(tmp_image_dir, "blurry.jpg"))
        mod = DeblurModule(DEFAULT_CFG)
        rec = {"file_path": path}
        result = mod.process(path, rec)
        assert result["is_blurry"] == "是"
        assert result["bren_sharpness"] < DEFAULT_CFG["bren_threshold"]

    def test_records_laplacian_variance(self, tmp_image_dir):
        path = make_sharp_image(os.path.join(tmp_image_dir, "sharp2.jpg"))
        mod = DeblurModule(DEFAULT_CFG)
        result = mod.process(path, {"file_path": path})
        assert "laplacian_variance" in result
        assert result["laplacian_variance"] >= 0

    def test_disabled_module_skips_processing(self, tmp_image_dir):
        path = make_blurry_image(os.path.join(tmp_image_dir, "b.jpg"))
        mod = DeblurModule({**DEFAULT_CFG, "enabled": False})
        result = mod.process(path, {"file_path": path})
        assert "is_blurry" not in result
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_deblur.py -v
```

期望：`ImportError`

- [ ] **Step 3: 实现 modules/deblur.py**

```python
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from modules.base import BaseModule


def _compute_bren_sharpness(gray: np.ndarray) -> Tuple[float, float]:
    """
    BREN = sum((I(x+2,y) - I(x,y))^2) / (H*W)
    拉普拉斯方差 = var(Laplacian(gray))
    """
    g = gray.astype(np.float64)
    diff = g[:, :-2] - g[:, 2:]
    bren = float(np.sum(diff ** 2) / gray.size)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    return bren, lap_var


class DeblurModule(BaseModule):
    """基于 BREN 锐度的模糊检测模块"""

    @property
    def name(self) -> str:
        return "deblur"

    def process(self, image_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return record

        resize: Optional[List[int]] = self.config.get("resize_for_analysis")
        threshold: float = self.config.get("bren_threshold", 50)

        img = cv2.imread(image_path)
        if img is None:
            record["is_blurry"] = "未知"
            record["bren_sharpness"] = -1.0
            record["laplacian_variance"] = -1.0
            return record

        if resize:
            img = cv2.resize(img, tuple(resize))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bren, lap_var = _compute_bren_sharpness(gray)

        record["bren_sharpness"] = round(bren, 4)
        record["laplacian_variance"] = round(lap_var, 4)
        record["is_blurry"] = "是" if bren < threshold else "否"
        return record
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_deblur.py -v
```

期望：4 passed。

- [ ] **Step 5: Commit**

```bash
git add modules/deblur.py tests/test_deblur.py
git commit -m "feat: deblur module with BREN sharpness detection"
```

---

## Task 9: modules/anomaly_detection.py

**Files:**
- Create: `modules/anomaly_detection.py`
- Create: `tests/test_anomaly_detection.py`

- [ ] **Step 1: 写失败测试**

`tests/test_anomaly_detection.py`:
```python
import pytest
import numpy as np
import cv2
import os
from modules.anomaly_detection import AnomalyDetectionModule


DEFAULT_CFG = {
    "enabled": True,
    "overexposed_threshold": 240,
    "underexposed_threshold": 15,
    "all_black_threshold": 5,
    "all_white_threshold": 250,
    "low_entropy_threshold": 2,
    "low_saturation_threshold": 10,
}


def save_img(arr: np.ndarray, path: str) -> str:
    cv2.imwrite(path, arr)
    return path


class TestAnomalyDetection:
    def test_normal_image_no_anomaly(self, tmp_image_dir):
        img = np.random.randint(80, 180, (100, 100, 3), dtype=np.uint8)
        path = save_img(img, os.path.join(tmp_image_dir, "normal.jpg"))
        mod = AnomalyDetectionModule(DEFAULT_CFG)
        result = mod.process(path, {})
        assert result["anomaly_type"] == "正常"

    def test_all_black_detected(self, tmp_image_dir):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = save_img(img, os.path.join(tmp_image_dir, "black.jpg"))
        mod = AnomalyDetectionModule(DEFAULT_CFG)
        result = mod.process(path, {})
        assert result["anomaly_type"] == "全黑"

    def test_overexposed_detected(self, tmp_image_dir):
        img = np.full((100, 100, 3), 250, dtype=np.uint8)
        path = save_img(img, os.path.join(tmp_image_dir, "bright.jpg"))
        mod = AnomalyDetectionModule(DEFAULT_CFG)
        result = mod.process(path, {})
        assert result["anomaly_type"] in ("过曝", "全白")

    def test_underexposed_detected(self, tmp_image_dir):
        img = np.full((100, 100, 3), 10, dtype=np.uint8)
        path = save_img(img, os.path.join(tmp_image_dir, "dark.jpg"))
        mod = AnomalyDetectionModule(DEFAULT_CFG)
        result = mod.process(path, {})
        assert result["anomaly_type"] in ("欠曝", "全黑")

    def test_records_all_fields(self, tmp_image_dir):
        img = np.random.randint(80, 180, (100, 100, 3), dtype=np.uint8)
        path = save_img(img, os.path.join(tmp_image_dir, "n2.jpg"))
        mod = AnomalyDetectionModule(DEFAULT_CFG)
        result = mod.process(path, {})
        for field in ["brightness_mean", "brightness_std", "entropy", "saturation_mean", "anomaly_type"]:
            assert field in result

    def test_disabled_module_skips(self, tmp_image_dir):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = save_img(img, os.path.join(tmp_image_dir, "skip.jpg"))
        mod = AnomalyDetectionModule({**DEFAULT_CFG, "enabled": False})
        result = mod.process(path, {})
        assert "anomaly_type" not in result
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_anomaly_detection.py -v
```

期望：`ImportError`

- [ ] **Step 3: 实现 modules/anomaly_detection.py**

```python
import cv2
import numpy as np
from typing import Dict, Any

from modules.base import BaseModule


def _compute_entropy(gray: np.ndarray) -> float:
    """基于归一化灰度直方图计算香农熵"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


class AnomalyDetectionModule(BaseModule):
    """基于亮度/熵/HSV饱和度的异常图片检测模块"""

    @property
    def name(self) -> str:
        return "anomaly_detection"

    def process(self, image_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return record

        cfg = self.config
        img = cv2.imread(image_path)
        if img is None:
            record["brightness_mean"] = -1.0
            record["brightness_std"] = -1.0
            record["entropy"] = -1.0
            record["saturation_mean"] = -1.0
            record["anomaly_type"] = "读取失败"
            return record

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
        s_channel = hsv[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b_mean = float(v_channel.mean())
        b_std = float(v_channel.std())
        entropy = _compute_entropy(gray)
        s_mean = float(s_channel.mean())

        record["brightness_mean"] = round(b_mean, 2)
        record["brightness_std"] = round(b_std, 2)
        record["entropy"] = round(entropy, 4)
        record["saturation_mean"] = round(s_mean, 2)

        # 判断异常类型（优先级由高到低）
        anomaly = "正常"
        if b_mean < cfg["all_black_threshold"] and b_std < 5:
            anomaly = "全黑"
        elif b_mean > cfg["all_white_threshold"] and b_std < 5:
            anomaly = "全白"
        elif b_mean > cfg["overexposed_threshold"]:
            anomaly = "过曝"
        elif b_mean < cfg["underexposed_threshold"]:
            anomaly = "欠曝"
        elif entropy < cfg["low_entropy_threshold"]:
            anomaly = "低熵（接近纯色）"
        elif s_mean < cfg["low_saturation_threshold"]:
            anomaly = "低饱和度（灰度图）"

        record["anomaly_type"] = anomaly
        return record
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_anomaly_detection.py -v
```

期望：6 passed。

- [ ] **Step 5: Commit**

```bash
git add modules/anomaly_detection.py tests/test_anomaly_detection.py
git commit -m "feat: anomaly detection with brightness/entropy/saturation checks"
```

---

## Task 10: modules/deduplication.py

**Files:**
- Create: `modules/deduplication.py`
- Create: `tests/test_deduplication.py`

- [ ] **Step 1: 写失败测试**

`tests/test_deduplication.py`:
```python
import pytest
import numpy as np
import cv2
import os
from modules.deduplication import DeduplicationModule


DEFAULT_CFG = {
    "enabled": True,
    "phash_hash_size": 16,
    "phash_threshold": 10,
    "ssim_threshold": 0.95,
    "ssim_win_size": 7,
    "use_phash_prefilter": True,
}


def save_img(arr: np.ndarray, path: str) -> str:
    cv2.imwrite(path, arr)
    return path


class TestDeduplicationModule:
    def test_identical_images_are_duplicates(self, tmp_image_dir):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        p1 = save_img(img, os.path.join(tmp_image_dir, "img1.jpg"))
        p2 = save_img(img, os.path.join(tmp_image_dir, "img2.jpg"))
        mod = DeduplicationModule(DEFAULT_CFG)
        results = mod.process_batch([p1, p2])
        kept = [r for r in results if r["is_duplicate"] == "否"]
        dupes = [r for r in results if r["is_duplicate"] == "是"]
        assert len(kept) == 1
        assert len(dupes) == 1
        assert dupes[0]["duplicate_of"] == os.path.basename(p1)

    def test_different_images_both_kept(self, tmp_image_dir):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        p1 = save_img(img1, os.path.join(tmp_image_dir, "dark.jpg"))
        p2 = save_img(img2, os.path.join(tmp_image_dir, "bright.jpg"))
        mod = DeduplicationModule(DEFAULT_CFG)
        results = mod.process_batch([p1, p2])
        assert all(r["is_duplicate"] == "否" for r in results)

    def test_single_image_not_duplicate(self, tmp_image_dir):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        p = save_img(img, os.path.join(tmp_image_dir, "solo.jpg"))
        mod = DeduplicationModule(DEFAULT_CFG)
        results = mod.process_batch([p])
        assert results[0]["is_duplicate"] == "否"
        assert results[0]["duplicate_of"] == ""

    def test_records_phash_field(self, tmp_image_dir):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        p = save_img(img, os.path.join(tmp_image_dir, "h.jpg"))
        mod = DeduplicationModule(DEFAULT_CFG)
        results = mod.process_batch([p])
        assert "phash" in results[0]
        assert results[0]["phash"] != ""

    def test_disabled_module_marks_all_not_duplicate(self, tmp_image_dir):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        p1 = save_img(img, os.path.join(tmp_image_dir, "x1.jpg"))
        p2 = save_img(img, os.path.join(tmp_image_dir, "x2.jpg"))
        mod = DeduplicationModule({**DEFAULT_CFG, "enabled": False})
        results = mod.process_batch([p1, p2])
        assert all(r["is_duplicate"] == "否" for r in results)
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_deduplication.py -v
```

期望：`ImportError`

- [ ] **Step 3: 实现 modules/deduplication.py**

```python
import cv2
import numpy as np
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Any, List, Tuple

from modules.base import BaseModule


def _compute_phash(image_path: str, hash_size: int) -> str:
    with Image.open(image_path) as img:
        return str(imagehash.phash(img, hash_size=hash_size))


def _hamming_distance(h1: str, h2: str) -> int:
    """计算两个十六进制字符串表示的 pHash 汉明距离"""
    n = int(h1, 16) ^ int(h2, 16)
    return bin(n).count("1")


def _compute_ssim(path1: str, path2: str, win_size: int) -> float:
    """读取两张图片并计算 SSIM（转为灰度，统一尺寸）"""
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return 0.0
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    score, _ = ssim(img1, img2, full=True, win_size=win_size)
    return float(score)


class DeduplicationModule(BaseModule):
    """
    两阶段去重：pHash 预筛选 + SSIM 精比对。
    与其他模块不同，去重需要全批次数据，因此提供 process_batch 接口。
    """

    @property
    def name(self) -> str:
        return "deduplication"

    def process(self, image_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """单张处理接口（兼容 BaseModule）；实际去重逻辑在 process_batch。"""
        record.setdefault("phash", "")
        record.setdefault("phash_hamming_distance", -1)
        record.setdefault("ssim_score", -1.0)
        record.setdefault("is_duplicate", "否")
        record.setdefault("duplicate_of", "")
        return record

    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        对整批图片执行去重，返回每张图片的记录字典列表。
        """
        import os
        records: List[Dict[str, Any]] = []
        for p in image_paths:
            records.append({
                "file_path": p,
                "phash": "",
                "phash_hamming_distance": -1,
                "ssim_score": -1.0,
                "is_duplicate": "否",
                "duplicate_of": "",
            })

        if not self.enabled:
            return records

        hash_size: int = self.config.get("phash_hash_size", 16)
        phash_thr: int = self.config.get("phash_threshold", 10)
        ssim_thr: float = self.config.get("ssim_threshold", 0.95)
        win_size: int = self.config.get("ssim_win_size", 7)
        use_prefilter: bool = self.config.get("use_phash_prefilter", True)

        # 阶段一：计算所有 pHash
        for rec in records:
            try:
                rec["phash"] = _compute_phash(rec["file_path"], hash_size)
            except Exception:
                rec["phash"] = ""

        # 已确认保留的图片集合（file_path -> phash）
        kept: List[Dict] = []

        for i, rec in enumerate(records):
            if rec["phash"] == "":
                continue  # 读取失败，跳过

            duplicate_found = False
            for kept_rec in kept:
                if kept_rec["phash"] == "":
                    continue

                # pHash 预过滤
                if use_prefilter:
                    hd = _hamming_distance(rec["phash"], kept_rec["phash"])
                    if hd > phash_thr:
                        continue
                    rec["phash_hamming_distance"] = hd

                # SSIM 精比对
                score = _compute_ssim(rec["file_path"], kept_rec["file_path"], win_size)
                rec["ssim_score"] = round(score, 4)
                if score >= ssim_thr:
                    rec["is_duplicate"] = "是"
                    rec["duplicate_of"] = os.path.basename(kept_rec["file_path"])
                    duplicate_found = True
                    break

            if not duplicate_found:
                kept.append(rec)

        return records
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_deduplication.py -v
```

期望：5 passed。

- [ ] **Step 5: 更新 modules/__init__.py**

```python
from .deduplication import DeduplicationModule
from .deblur import DeblurModule
from .anomaly_detection import AnomalyDetectionModule

__all__ = ["DeduplicationModule", "DeblurModule", "AnomalyDetectionModule"]
```

- [ ] **Step 6: Commit**

```bash
git add modules/deduplication.py modules/__init__.py tests/test_deduplication.py
git commit -m "feat: deduplication module with pHash prefilter + SSIM verification"
```

---

## Task 11: config.yaml

**Files:**
- Create: `config.yaml`

- [ ] **Step 1: 创建 config.yaml**

```yaml
# 模块启用/禁用配置
modules:
  deduplication:
    enabled: true
    phash_hash_size: 16
    phash_threshold: 10
    ssim_threshold: 0.95
    ssim_win_size: 7
    use_phash_prefilter: true

  deblur:
    enabled: true
    bren_threshold: 50
    resize_for_analysis: [512, 512]

  anomaly_detection:
    enabled: true
    overexposed_threshold: 240
    underexposed_threshold: 15
    all_black_threshold: 5
    all_white_threshold: 250
    low_entropy_threshold: 2
    low_saturation_threshold: 10

# 并发配置
concurrency:
  enabled: true
  num_workers: 0        # 0 = 自动检测 CPU 核心数
  batch_size: 100
  memory_limit_mb: 2048

# 进度显示
progress:
  enabled: true
  update_interval: 10

# 文件处理
file:
  image_extensions: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]
  recursive: false
  output_dir: "output"
  csv_encoding: "utf-8-sig"

# 日志
logging:
  level: INFO
  max_bytes: 10485760   # 10MB
  backup_count: 3
  format: "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
```

- [ ] **Step 2: 验证配置可加载**

```bash
python -c "from utils.config_loader import load_config; cfg=load_config('config.yaml'); print('ok')"
```

期望：输出 `ok`。

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "feat: default config.yaml"
```

---

## Task 12: main.py

**Files:**
- Create: `main.py`

- [ ] **Step 1: 实现 main.py**

```python
"""
图片批量处理工具 - 主入口

用法：
  python main.py -i /path/to/images -o output -c config.yaml
  python main.py -i /path/to/images --no-dedup --no-deblur
  python main.py --dry-run
"""

import argparse
import os
import shutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.memory_monitor import MemoryMonitor
from utils.progress import ProgressTracker
from utils.file_handler import scan_images, validate_image, create_output_dirs, write_csv
from modules.deduplication import DeduplicationModule
from modules.deblur import DeblurModule
from modules.anomaly_detection import AnomalyDetectionModule

# CSV 列名顺序（与解决方案设计一致）
CSV_FIELDS = [
    "filename", "file_path", "file_size_kb", "image_width", "image_height",
    "phash", "phash_hamming_distance", "ssim_score", "is_duplicate", "duplicate_of",
    "bren_sharpness", "laplacian_variance", "is_blurry",
    "brightness_mean", "brightness_std", "entropy", "saturation_mean", "anomaly_type",
    "final_keep", "reject_reason", "process_time_ms",
]


def parse_args():
    parser = argparse.ArgumentParser(description="图片批量处理工具")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("-i", "--input", default=".", help="输入目录路径")
    parser.add_argument("-o", "--output", default=None, help="输出目录路径（默认来自配置文件）")
    parser.add_argument("--no-dedup", action="store_true", help="禁用去重模块")
    parser.add_argument("--no-deblur", action="store_true", help="禁用去模糊模块")
    parser.add_argument("--no-anomaly", action="store_true", help="禁用异常检测模块")
    parser.add_argument("-w", "--workers", type=int, default=None, help="工作进程数（0=自动检测）")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不保存文件")
    return parser.parse_args()


def _build_base_record(image_path: str) -> Dict[str, Any]:
    """构建图片基础信息字典"""
    from PIL import Image
    stat = os.stat(image_path)
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        width, height = -1, -1
    return {
        "filename": os.path.basename(image_path),
        "file_path": image_path,
        "file_size_kb": round(stat.st_size / 1024, 2),
        "image_width": width,
        "image_height": height,
        "phash": "",
        "phash_hamming_distance": -1,
        "ssim_score": -1.0,
        "is_duplicate": "否",
        "duplicate_of": "",
        "bren_sharpness": -1.0,
        "laplacian_variance": -1.0,
        "is_blurry": "否",
        "brightness_mean": -1.0,
        "brightness_std": -1.0,
        "entropy": -1.0,
        "saturation_mean": -1.0,
        "anomaly_type": "正常",
        "final_keep": "是",
        "reject_reason": "",
        "process_time_ms": 0.0,
    }


def _process_single_image(image_path: str, deblur_cfg: dict, anomaly_cfg: dict) -> Dict[str, Any]:
    """子进程中处理单张图片（去模糊 + 异常检测）"""
    t0 = time.time()
    record = _build_base_record(image_path)
    deblur_mod = DeblurModule(deblur_cfg)
    record = deblur_mod.process(image_path, record)
    anomaly_mod = AnomalyDetectionModule(anomaly_cfg)
    record = anomaly_mod.process(image_path, record)
    record["process_time_ms"] = round((time.time() - t0) * 1000, 2)
    return record


def _determine_final_keep(record: Dict[str, Any]) -> Dict[str, Any]:
    """根据各模块结果设置 final_keep 和 reject_reason"""
    reasons = []
    if record.get("is_duplicate") == "是":
        reasons.append(f"重复（与 {record.get('duplicate_of', '')} 相同）")
    if record.get("is_blurry") == "是":
        reasons.append("模糊")
    anomaly = record.get("anomaly_type", "正常")
    if anomaly != "正常":
        reasons.append(f"异常：{anomaly}")
    if reasons:
        record["final_keep"] = "否"
        record["reject_reason"] = "；".join(reasons)
    else:
        record["final_keep"] = "是"
        record["reject_reason"] = ""
    return record


def _ensure_ordered_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """保证 CSV 列顺序，缺失字段填空"""
    return {f: record.get(f, "") for f in CSV_FIELDS}


def process_folder(
    folder_path: str,
    cfg: dict,
    output_base: str,
    dry_run: bool,
    logger,
) -> Dict[str, int]:
    """处理单个子文件夹，返回统计信息"""
    file_cfg = cfg["file"]
    conc_cfg = cfg["concurrency"]
    dedup_cfg = cfg["modules"]["deduplication"]
    deblur_cfg = cfg["modules"]["deblur"]
    anomaly_cfg = cfg["modules"]["anomaly_detection"]
    prog_cfg = cfg["progress"]

    folder_name = os.path.basename(folder_path)
    image_paths = scan_images(folder_path, file_cfg["image_extensions"], file_cfg["recursive"])

    if not image_paths:
        logger.warning(f"[{folder_name}] 无图片，跳过")
        return {"total": 0, "kept": 0, "rejected": 0}

    logger.info(f"[{folder_name}] 发现 {len(image_paths)} 张图片")

    # 步骤一：去重（需要整批）
    dedup_mod = DeduplicationModule(dedup_cfg)
    dedup_results = dedup_mod.process_batch(image_paths)
    # 建立 path -> dedup_record 映射
    dedup_map = {r["file_path"]: r for r in dedup_results}

    # 步骤二：去模糊 + 异常检测（并行）
    num_workers = conc_cfg.get("num_workers") or multiprocessing.cpu_count()
    batch_size = conc_cfg.get("batch_size", 100)
    memory_monitor = MemoryMonitor(conc_cfg.get("memory_limit_mb", 2048))
    all_records: List[Dict[str, Any]] = []

    with ProgressTracker(len(image_paths), f"处理 {folder_name}", prog_cfg.get("enabled", True)) as prog:
        for batch_start in range(0, len(image_paths), batch_size):
            batch = image_paths[batch_start: batch_start + batch_size]

            if conc_cfg.get("enabled", True) and num_workers > 1:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_process_single_image, p, deblur_cfg, anomaly_cfg): p
                        for p in batch
                    }
                    for future in as_completed(futures):
                        try:
                            rec = future.result()
                        except Exception as e:
                            p = futures[future]
                            logger.error(f"处理失败: {p}, 错误: {e}")
                            rec = _build_base_record(p)
                        # 合并去重结果
                        dedup_rec = dedup_map.get(rec["file_path"], {})
                        rec.update({
                            "phash": dedup_rec.get("phash", ""),
                            "phash_hamming_distance": dedup_rec.get("phash_hamming_distance", -1),
                            "ssim_score": dedup_rec.get("ssim_score", -1.0),
                            "is_duplicate": dedup_rec.get("is_duplicate", "否"),
                            "duplicate_of": dedup_rec.get("duplicate_of", ""),
                        })
                        all_records.append(rec)
                        prog.update()
            else:
                for p in batch:
                    try:
                        rec = _process_single_image(p, deblur_cfg, anomaly_cfg)
                    except Exception as e:
                        logger.error(f"处理失败: {p}, 错误: {e}")
                        rec = _build_base_record(p)
                    dedup_rec = dedup_map.get(p, {})
                    rec.update({
                        "phash": dedup_rec.get("phash", ""),
                        "phash_hamming_distance": dedup_rec.get("phash_hamming_distance", -1),
                        "ssim_score": dedup_rec.get("ssim_score", -1.0),
                        "is_duplicate": dedup_rec.get("is_duplicate", "否"),
                        "duplicate_of": dedup_rec.get("duplicate_of", ""),
                    })
                    all_records.append(rec)
                    prog.update()

            memory_monitor.check_and_collect()

    # 步骤三：确定最终保留/拒绝
    for rec in all_records:
        _determine_final_keep(rec)

    # 步骤四：输出
    if not dry_run:
        filtered_dir, csv_path = create_output_dirs(output_base, folder_name)
        for rec in all_records:
            if rec["final_keep"] == "是":
                dst = os.path.join(filtered_dir, rec["filename"])
                shutil.copy2(rec["file_path"], dst)
        ordered_records = [_ensure_ordered_fields(r) for r in all_records]
        write_csv(ordered_records, csv_path, file_cfg["csv_encoding"])
        logger.info(f"[{folder_name}] 报告已写入: {csv_path}")

    kept = sum(1 for r in all_records if r["final_keep"] == "是")
    rejected = len(all_records) - kept
    logger.info(f"[{folder_name}] 保留 {kept} 张，拒绝 {rejected} 张")
    return {"total": len(all_records), "kept": kept, "rejected": rejected}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 命令行参数覆盖配置
    if args.no_dedup:
        cfg["modules"]["deduplication"]["enabled"] = False
    if args.no_deblur:
        cfg["modules"]["deblur"]["enabled"] = False
    if args.no_anomaly:
        cfg["modules"]["anomaly_detection"]["enabled"] = False
    if args.workers is not None:
        cfg["concurrency"]["num_workers"] = args.workers
    if args.verbose:
        cfg["logging"]["level"] = "DEBUG"

    output_base = args.output or cfg["file"]["output_dir"]
    logger = setup_logger("main", config=cfg["logging"])

    if args.dry_run:
        logger.info("=== 试运行模式，不保存文件 ===")

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return 1

    # 扫描子文件夹
    subfolders = [
        os.path.join(input_dir, d)
        for d in sorted(os.listdir(input_dir))
        if os.path.isdir(os.path.join(input_dir, d)) and d not in ("output", "logs")
    ]

    if not subfolders:
        logger.warning(f"输入目录下无子文件夹: {input_dir}，将直接处理输入目录本身")
        subfolders = [input_dir]

    total_stats = {"total": 0, "kept": 0, "rejected": 0}
    for folder in subfolders:
        stats = process_folder(folder, cfg, output_base, args.dry_run, logger)
        for k in total_stats:
            total_stats[k] += stats[k]

    logger.info(
        f"=== 完成 === 总计 {total_stats['total']} 张，"
        f"保留 {total_stats['kept']} 张，拒绝 {total_stats['rejected']} 张"
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 2: 验证语法无误**

```bash
python -m py_compile main.py && echo "syntax ok"
```

期望：输出 `syntax ok`。

- [ ] **Step 3: 运行全部测试确认无回归**

```bash
pytest tests/ -v
```

期望：全部通过。

- [ ] **Step 4: 端到端冒烟测试（dry-run）**

```bash
mkdir -p tmp_smoke/folder1
python -c "
import cv2, numpy as np, os
for i in range(3):
    img = np.random.randint(0,255,(100,100,3),dtype=np.uint8)
    cv2.imwrite(f'tmp_smoke/folder1/img{i}.jpg', img)
"
python main.py -i tmp_smoke --dry-run
```

期望：输出处理日志，无异常，不生成 output 目录。

- [ ] **Step 5: 清理冒烟测试目录**

```bash
rm -rf tmp_smoke
```

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: main entry with parallel processing, CLI args, dry-run support"
```

---

## Task 13: 最终验收

- [ ] **Step 1: 运行全量测试**

```bash
pytest tests/ -v --tb=short
```

期望：全部通过，0 failures。

- [ ] **Step 2: 验证配置加载和导入链**

```bash
python -c "
from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.memory_monitor import MemoryMonitor
from utils.progress import ProgressTracker
from utils.file_handler import scan_images, validate_image, write_csv
from modules import DeduplicationModule, DeblurModule, AnomalyDetectionModule
print('all imports ok')
"
```

期望：输出 `all imports ok`。

- [ ] **Step 3: 完整端到端测试（含输出验证）**

```bash
mkdir -p tmp_e2e/folder1 tmp_e2e/folder2

# 生成测试图片
python -c "
import cv2, numpy as np, os

# folder1: 3张正常图 + 1张模糊图
for i in range(3):
    img = np.random.randint(50,200,(100,100,3),dtype=np.uint8)
    cv2.imwrite(f'tmp_e2e/folder1/normal{i}.jpg', img)
blur = np.full((100,100,3),128,dtype=np.uint8)
blur = cv2.GaussianBlur(blur,(51,51),20)
cv2.imwrite('tmp_e2e/folder1/blurry.jpg', blur)

# folder2: 2张相同图片（重复）
img = np.random.randint(0,255,(100,100,3),dtype=np.uint8)
cv2.imwrite('tmp_e2e/folder2/dup1.jpg', img)
cv2.imwrite('tmp_e2e/folder2/dup2.jpg', img)
"

python main.py -i tmp_e2e -o tmp_e2e_output
```

验证输出：
```bash
# folder1 应保留 3 张（排除模糊）
ls tmp_e2e_output/folder1/filtered_images/ | wc -l

# folder2 应保留 1 张（排除重复）
ls tmp_e2e_output/folder2/filtered_images/ | wc -l

# CSV 应存在
ls tmp_e2e_output/folder1/analysis_report.csv
ls tmp_e2e_output/folder2/analysis_report.csv
```

期望：folder1 保留约 3 张，folder2 保留 1 张，两个 CSV 均存在。

- [ ] **Step 4: 清理测试目录**

```bash
rm -rf tmp_e2e tmp_e2e_output
```

- [ ] **Step 5: 最终 Commit**

```bash
git add .
git commit -m "chore: final integration verified, all tests passing"
```

---

## 自查结果

**Spec 覆盖检查：**
- ✅ 去重模块（pHash + SSIM）→ Task 10
- ✅ 去模糊模块（BREN）→ Task 8
- ✅ 异常检测（熵+亮度+HSV）→ Task 9
- ✅ 并发处理（ProcessPoolExecutor）→ Task 12
- ✅ 内存监控 + GC → Task 4
- ✅ 进度条（tqdm）→ Task 5
- ✅ 日志管理（轮转）→ Task 3
- ✅ CSV 输出（所有21列）→ Task 6 + Task 12
- ✅ 命令行参数（argparse）→ Task 12
- ✅ 配置文件（YAML，深度合并）→ Task 2 + Task 11
- ✅ 图片验证（PIL verify+load）→ Task 6
- ✅ 输出目录结构（output/folder/filtered_images/）→ Task 6

**类型一致性：**
- `DeduplicationModule.process_batch()` 返回 `List[Dict]`，Task 12 main 中正确调用
- `_process_single_image` 签名与 `ProcessPoolExecutor.submit` 调用一致
- CSV 字段名在 `CSV_FIELDS` 常量中统一定义，`write_csv` 使用相同字段名
