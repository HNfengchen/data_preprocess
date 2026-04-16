import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from modules.base import BaseModule


def _compute_bren_sharpness(gray: np.ndarray) -> Tuple[float, float]:
    """
    BREN = sum((I(x+2,y) - I(x,y))^2) / (H*W)
    Laplacian variance = var(Laplacian(gray))
    """
    g = gray.astype(np.float64)
    diff = g[:, :-2] - g[:, 2:]
    bren = float(np.sum(diff ** 2) / gray.size)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    return bren, lap_var


class DeblurModule(BaseModule):
    """Blur detection via BREN sharpness"""

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
