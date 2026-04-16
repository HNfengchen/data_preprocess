# 图片批量处理工具

对图片集进行**去重**、**去模糊**、**异常检测**，输出筛选后的图片和 CSV 分析报告。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 处理图片（输入目录需包含子文件夹，每个子文件夹单独处理）
python main.py -i /path/to/images

# 指定输出目录
python main.py -i /path/to/images -o /path/to/output
```

## 目录结构要求

输入目录应包含子文件夹，每个子文件夹作为一个批次独立处理：

```
input_dir/
├── folder1/        ← 每个子文件夹单独处理
│   ├── img001.jpg
│   └── img002.png
└── folder2/
    └── img003.jpg
```

输出结构：
```
output/
├── folder1/
│   ├── filtered_images/   ← 通过筛选的图片
│   └── analysis_report.csv
└── folder2/
    ├── filtered_images/
    └── analysis_report.csv
```

> **直接运行 `python main.py`（不加参数）**：默认扫描当前目录的子文件夹。如果当前目录是项目根目录，则扫描 `modules/`、`tests/` 等子目录，不含图片，输出若干 warning 后正常退出，**不会报错也不会损坏任何文件**。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --input` | `.`（当前目录）| 输入目录路径 |
| `-o, --output` | `output` | 输出目录路径 |
| `-c, --config` | `config.yaml` | 配置文件路径 |
| `-w, --workers` | 自动检测 CPU 核心数 | 并行工作进程数（0=自动）|
| `-v, --verbose` | 否 | 输出 DEBUG 日志 |
| `--no-dedup` | 否 | 禁用去重模块 |
| `--no-deblur` | 否 | 禁用去模糊模块 |
| `--no-anomaly` | 否 | 禁用异常检测模块 |
| `--dry-run` | 否 | 试运行，不保存任何文件 |

示例：

```bash
# 只做去重，不检测模糊和异常
python main.py -i ./images --no-deblur --no-anomaly

# 试运行：只看统计，不输出文件
python main.py -i ./images --dry-run

# 单进程运行（方便调试）
python main.py -i ./images -w 1 -v
```

## 处理流程

```
输入图片
  │
  ▼
去重（pHash 预筛 + SSIM 精比对）
  │
  ▼
去模糊（BREN 锐度评估）
  │
  ▼
异常检测（亮度 / 熵 / HSV 饱和度）
  │
  ▼
输出筛选图片 + CSV 报告
```

## 配置文件（config.yaml）

所有阈值均可在 `config.yaml` 中调整，无需修改代码。

**去重模块关键参数：**
- `phash_threshold`：pHash 汉明距离阈值，越小越严格（默认 10）
- `ssim_threshold`：SSIM 相似度阈值，越大越严格（默认 0.95）

**去模糊模块关键参数：**
- `bren_threshold`：BREN 锐度阈值，低于此值视为模糊（默认 50；范围建议 20–100）

**异常检测关键参数：**
- `overexposed_threshold`：亮度均值超过此值视为过曝（默认 240）
- `underexposed_threshold`：亮度均值低于此值视为欠曝（默认 15）
- `low_entropy_threshold`：熵低于此值视为纯色（默认 2）

## CSV 报告字段

| 字段 | 说明 |
|------|------|
| `filename` | 文件名 |
| `final_keep` | 是否保留（是/否）|
| `reject_reason` | 拒绝原因 |
| `is_duplicate` | 是否重复 |
| `duplicate_of` | 与哪张图片重复 |
| `is_blurry` | 是否模糊 |
| `bren_sharpness` | BREN 锐度值 |
| `anomaly_type` | 异常类型（正常/全黑/全白/过曝/欠曝等）|
| `brightness_mean` | 亮度均值 |
| `entropy` | 图像熵值 |
| `process_time_ms` | 单张处理耗时（ms）|

## 依赖

```
opencv-python   图像读取与处理
Pillow          图像验证与格式支持
scikit-image    SSIM 计算
imagehash       pHash 感知哈希
NumPy           数值计算
PyYAML          配置文件解析
tqdm            进度条
psutil          内存监控
```
