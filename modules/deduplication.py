import os
import logging
import cv2
import numpy as np
import imagehash
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Any, List, Optional, Tuple

from modules.base import BaseModule

# SSIM 计算时统一缩放到此尺寸，避免每次读原始大图
_SSIM_SIZE = (256, 256)


def _compute_phash(image_path: str, hash_size: int) -> str:
    with Image.open(image_path) as img:
        return str(imagehash.phash(img, hash_size=hash_size))


def _hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex-encoded pHash values"""
    n = int(h1, 16) ^ int(h2, 16)
    return bin(n).count("1")


def _load_thumbnail(image_path: str) -> Optional[np.ndarray]:
    """读取图片并缩放到固定小尺寸灰度图，用于 SSIM 缓存"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, _SSIM_SIZE)


def _ssim_from_thumbnails(t1: np.ndarray, t2: np.ndarray, win_size: int) -> float:
    """基于已缓存缩略图计算 SSIM，无磁盘 I/O"""
    score, _ = ssim(t1, t2, full=True, win_size=win_size)
    return float(score)


class DeduplicationModule(BaseModule):
    """Two-stage deduplication: pHash prefilter + SSIM verification

    优化点：
    1. 预计算所有图片的 256×256 灰度缩略图，每张只读一次磁盘
    2. 内层比对循环用 ThreadPoolExecutor 并行，多核同时计算 SSIM
    3. 每处理 1% 输出一次进度日志
    """

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

    def process_batch(self, image_paths: List[str], logger=None) -> List[Dict[str, Any]]:
        """对整批图片去重，返回每张图片的记录字典列表"""
        log = logger or logging.getLogger(__name__)

        records: List[Dict[str, Any]] = [
            {
                "file_path": p,
                "phash": "",
                "phash_hamming_distance": -1,
                "ssim_score": -1.0,
                "is_duplicate": "否",
                "duplicate_of": "",
            }
            for p in image_paths
        ]

        if not self.enabled:
            log.info("[dedup] 模块已禁用，跳过去重")
            return records

        n = len(records)
        hash_size: int = self.config.get("phash_hash_size", 16)
        phash_thr: int = self.config.get("phash_threshold", 10)
        ssim_thr: float = self.config.get("ssim_threshold", 0.95)
        win_size: int = self.config.get("ssim_win_size", 7)
        use_prefilter: bool = self.config.get("use_phash_prefilter", True)
        # 内层并行比对的线程数（不宜过多，SSIM 已是纯 CPU 计算）
        inner_workers: int = self.config.get("ssim_workers", min(4, os.cpu_count() or 4))

        log_step = max(1, n // 100)  # 每 1% 记录一次进度

        # ── 阶段1：并行计算所有 pHash ──────────────────────────────────
        log.info(f"[dedup] 阶段1/3 计算 pHash，共 {n} 张，hash_size={hash_size}")
        with ThreadPoolExecutor(max_workers=inner_workers) as ex:
            future_to_idx = {
                ex.submit(_compute_phash, rec["file_path"], hash_size): idx
                for idx, rec in enumerate(records)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    records[idx]["phash"] = future.result()
                except Exception as e:
                    log.warning(f"[dedup] pHash 失败: {os.path.basename(records[idx]['file_path'])}: {e}")
                    records[idx]["phash"] = ""
                done += 1
                if done % log_step == 0 or done == n:
                    log.info(f"[dedup] pHash {done}/{n} ({done*100//n}%)")

        # ── 阶段2：并行预计算 SSIM 缩略图（每张只读一次）────────────────
        log.info(f"[dedup] 阶段2/3 预加载缩略图 {_SSIM_SIZE}，共 {n} 张")
        thumbnails: Dict[str, Optional[np.ndarray]] = {}
        with ThreadPoolExecutor(max_workers=inner_workers) as ex:
            future_to_path = {
                ex.submit(_load_thumbnail, rec["file_path"]): rec["file_path"]
                for rec in records if rec["phash"]  # 跳过 pHash 失败的
            }
            done = 0
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    thumbnails[path] = future.result()
                except Exception as e:
                    log.warning(f"[dedup] 缩略图加载失败: {os.path.basename(path)}: {e}")
                    thumbnails[path] = None
                done += 1
                if done % log_step == 0 or done == n:
                    log.info(f"[dedup] 缩略图 {done}/{n} ({done*100//n}%)")

        # ── 阶段3：顺序遍历 + 内层并行 SSIM 比对 ────────────────────────
        log.info(f"[dedup] 阶段3/3 相似度比对，inner_workers={inner_workers}，ssim_thr={ssim_thr}")
        kept: List[Dict] = []   # 已确认保留的记录
        ssim_calls = 0

        with ThreadPoolExecutor(max_workers=inner_workers) as ex:
            for i, rec in enumerate(records):
                path = rec["file_path"]
                t_cur = thumbnails.get(path)

                if not rec["phash"] or t_cur is None:
                    # pHash 失败或缩略图加载失败 → 直接保留（不参与去重）
                    kept.append(rec)
                    if (i + 1) % log_step == 0 or (i + 1) == n:
                        log.info(f"[dedup] 比对 {i+1}/{n} ({(i+1)*100//n}%)，已保留 {len(kept)} 张，SSIM {ssim_calls} 次")
                    continue

                # 收集需要 SSIM 验证的候选（通过 pHash 预筛）
                candidates: List[Tuple[Dict, int]] = []  # (kept_rec, hamming_dist)
                for kept_rec in kept:
                    if not kept_rec["phash"]:
                        continue
                    if use_prefilter:
                        hd = _hamming_distance(rec["phash"], kept_rec["phash"])
                        if hd > phash_thr:
                            continue
                        candidates.append((kept_rec, hd))
                    else:
                        candidates.append((kept_rec, -1))

                if not candidates:
                    kept.append(rec)
                    if (i + 1) % log_step == 0 or (i + 1) == n:
                        log.info(f"[dedup] 比对 {i+1}/{n} ({(i+1)*100//n}%)，已保留 {len(kept)} 张，SSIM {ssim_calls} 次")
                    continue

                # 并行 SSIM：当前图 vs 所有候选
                ssim_calls += len(candidates)
                future_to_cand = {
                    ex.submit(
                        _ssim_from_thumbnails,
                        t_cur,
                        thumbnails.get(cand["file_path"]),
                        win_size
                    ): (cand, hd)
                    for cand, hd in candidates
                    if thumbnails.get(cand["file_path"]) is not None
                }

                duplicate_found = False
                best_score = -1.0
                best_cand = None
                for future in as_completed(future_to_cand):
                    cand_rec, hd = future_to_cand[future]
                    try:
                        score = future.result()
                    except Exception:
                        score = 0.0
                    if score > best_score:
                        best_score = score
                        best_cand = (cand_rec, hd)
                    if score >= ssim_thr and not duplicate_found:
                        rec["is_duplicate"] = "是"
                        rec["duplicate_of"] = os.path.basename(cand_rec["file_path"])
                        rec["ssim_score"] = round(score, 4)
                        rec["phash_hamming_distance"] = hd
                        duplicate_found = True
                        # 找到重复后不 break，让其余 future 也完成（避免泄漏）

                if not duplicate_found:
                    if best_cand:
                        rec["ssim_score"] = round(best_score, 4)
                        rec["phash_hamming_distance"] = best_cand[1]
                    kept.append(rec)

                if (i + 1) % log_step == 0 or (i + 1) == n:
                    dupes = i + 1 - len(kept)
                    log.info(
                        f"[dedup] 比对 {i+1}/{n} ({(i+1)*100//n}%)，"
                        f"已保留 {len(kept)} 张，重复 {dupes} 张，"
                        f"SSIM总调用 {ssim_calls} 次，候选 {len(candidates)} 个"
                    )

        total_dupes = n - len(kept)
        log.info(f"[dedup] 完成：保留 {len(kept)} 张，重复 {total_dupes} 张，SSIM总调用 {ssim_calls} 次")
        return records
