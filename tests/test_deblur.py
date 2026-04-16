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
    """Checkerboard = high gradient = high sharpness"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[::4, :] = 255
    img[:, ::4] = 255
    cv2.imwrite(path, img)
    return path


def make_blurry_image(path):
    """Solid color + heavy blur = low gradient = low sharpness"""
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
