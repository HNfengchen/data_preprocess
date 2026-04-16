#!/bin/bash
# 图片批量处理工具打包脚本（Linux）
# 用法：chmod +x build.sh && ./build.sh

set -e  # 任意步骤失败即退出

ENV_NAME="image_proc_pack"
PYTHON_VER="3.13"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# 初始化 conda（非交互式 shell 需要手动 source）
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo "ERROR: conda not found"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo ">>> [1/5] 创建打包环境: $ENV_NAME (Python $PYTHON_VER)"
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

echo ">>> [2/5] 激活环境"
conda activate "$ENV_NAME"

echo ">>> [3/5] 安装项目依赖（清华镜像）"
pip install -r requirements.txt -i "$MIRROR"

echo ">>> [4/5] 安装 PyInstaller"
pip install pyinstaller -i "$MIRROR"

echo ">>> [5/5] 打包为单文件可执行程序"
pyinstaller --onefile \
    --name image_processor \
    --add-data "config.yaml:." \
    --hidden-import "modules.deduplication" \
    --hidden-import "modules.deblur" \
    --hidden-import "modules.anomaly_detection" \
    --hidden-import "skimage.metrics" \
    main.py

echo ">>> 清理打包环境"
conda deactivate
conda env remove -n "$ENV_NAME" -y

echo ""
echo "=== 完成 ==="
echo "可执行文件: $(pwd)/dist/image_processor"
echo "用法: ./dist/image_processor -i /path/to/images -o output"
