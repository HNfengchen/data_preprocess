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
    """Hamming distance between two hex-encoded pHash values"""
    n = int(h1, 16) ^ int(h2, 16)
    return bin(n).count("1")


def _compute_ssim(path1: str, path2: str, win_size: int) -> float:
    """Load two images and compute SSIM (grayscale, unified size)"""
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
    """Two-stage deduplication: pHash prefilter + SSIM verification"""

    @property
    def name(self) -> str:
        return "deduplication"

    def process(self, image_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Single-image interface (compatibility). Actual dedup logic in process_batch."""
        record.setdefault("phash", "")
        record.setdefault("phash_hamming_distance", -1)
        record.setdefault("ssim_score", -1.0)
        record.setdefault("is_duplicate", "否")
        record.setdefault("duplicate_of", "")
        return record

    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process batch of images for deduplication."""
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

        # Stage 1: Compute all pHash values
        for rec in records:
            try:
                rec["phash"] = _compute_phash(rec["file_path"], hash_size)
            except Exception:
                rec["phash"] = ""

        # Track kept images
        kept: List[Dict] = []

        for i, rec in enumerate(records):
            if rec["phash"] == "":
                continue

            duplicate_found = False
            for kept_rec in kept:
                if kept_rec["phash"] == "":
                    continue

                # pHash prefilter
                if use_prefilter:
                    hd = _hamming_distance(rec["phash"], kept_rec["phash"])
                    if hd > phash_thr:
                        continue
                    rec["phash_hamming_distance"] = hd

                # SSIM verification
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
