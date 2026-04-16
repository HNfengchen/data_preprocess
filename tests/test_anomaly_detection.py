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
