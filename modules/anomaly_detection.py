import cv2
import numpy as np
from typing import Dict, Any

from modules.base import BaseModule


def _compute_entropy(gray: np.ndarray) -> float:
    """Shannon entropy based on normalized grayscale histogram"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


class AnomalyDetectionModule(BaseModule):
    """Anomaly detection via brightness/entropy/HSV saturation"""

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

        # Determine anomaly type (priority order)
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
