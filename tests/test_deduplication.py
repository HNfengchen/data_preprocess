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
