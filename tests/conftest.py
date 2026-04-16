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
