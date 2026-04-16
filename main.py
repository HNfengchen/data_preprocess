"""
Image batch processing tool - main entry point

Usage:
  python main.py -i /path/to/images -o output -c config.yaml
  python main.py -i /path/to/images --no-dedup --no-deblur
  python main.py --dry-run
"""

import argparse
import os
import shutil
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.memory_monitor import MemoryMonitor
from utils.progress import ProgressTracker
from utils.file_handler import scan_images, validate_image, create_output_dirs, write_csv
from modules.deduplication import DeduplicationModule
from modules.deblur import DeblurModule
from modules.anomaly_detection import AnomalyDetectionModule

# CSV column order (21 columns from spec)
CSV_FIELDS = [
    "filename", "file_path", "file_size_kb", "image_width", "image_height",
    "phash", "phash_hamming_distance", "ssim_score", "is_duplicate", "duplicate_of",
    "bren_sharpness", "laplacian_variance", "is_blurry",
    "brightness_mean", "brightness_std", "entropy", "saturation_mean", "anomaly_type",
    "final_keep", "reject_reason", "process_time_ms",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Image batch processing tool")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("-i", "--input", default=".", help="Input directory path")
    parser.add_argument("-o", "--output", default=None, help="Output directory path (from config if not specified)")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    parser.add_argument("--no-deblur", action="store_true", help="Disable deblur")
    parser.add_argument("--no-anomaly", action="store_true", help="Disable anomaly detection")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Worker count (0=auto)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no files saved)")
    return parser.parse_args()


def _build_base_record(image_path: str) -> Dict[str, Any]:
    """Build base image record with file info"""
    from PIL import Image
    stat = os.stat(image_path)
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        width, height = -1, -1
    return {
        "filename": os.path.basename(image_path),
        "file_path": image_path,
        "file_size_kb": round(stat.st_size / 1024, 2),
        "image_width": width,
        "image_height": height,
        "phash": "",
        "phash_hamming_distance": -1,
        "ssim_score": -1.0,
        "is_duplicate": "否",
        "duplicate_of": "",
        "bren_sharpness": -1.0,
        "laplacian_variance": -1.0,
        "is_blurry": "否",
        "brightness_mean": -1.0,
        "brightness_std": -1.0,
        "entropy": -1.0,
        "saturation_mean": -1.0,
        "anomaly_type": "正常",
        "final_keep": "是",
        "reject_reason": "",
        "process_time_ms": 0.0,
    }


def _process_single_image(image_path: str, deblur_cfg: dict, anomaly_cfg: dict) -> Dict[str, Any]:
    """Process single image (deblur + anomaly detection) in subprocess"""
    t0 = time.time()
    record = _build_base_record(image_path)
    deblur_mod = DeblurModule(deblur_cfg)
    record = deblur_mod.process(image_path, record)
    anomaly_mod = AnomalyDetectionModule(anomaly_cfg)
    record = anomaly_mod.process(image_path, record)
    record["process_time_ms"] = round((time.time() - t0) * 1000, 2)
    return record


def _determine_final_keep(record: Dict[str, Any]) -> Dict[str, Any]:
    """Determine final_keep and reject_reason based on all module results"""
    reasons = []
    if record.get("is_duplicate") == "是":
        reasons.append(f"duplicate (same as {record.get('duplicate_of', '')})")
    if record.get("is_blurry") == "是":
        reasons.append("blurry")
    anomaly = record.get("anomaly_type", "正常")
    if anomaly != "正常":
        reasons.append(f"anomaly: {anomaly}")
    if reasons:
        record["final_keep"] = "否"
        record["reject_reason"] = "; ".join(reasons)
    else:
        record["final_keep"] = "是"
        record["reject_reason"] = ""
    return record


def _ensure_ordered_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure CSV column order, fill missing fields with empty strings"""
    return {f: record.get(f, "") for f in CSV_FIELDS}


def process_folder(
    folder_path: str,
    cfg: dict,
    output_base: str,
    dry_run: bool,
    logger,
) -> Dict[str, int]:
    """Process single folder, return statistics"""
    file_cfg = cfg["file"]
    conc_cfg = cfg["concurrency"]
    dedup_cfg = cfg["modules"]["deduplication"]
    deblur_cfg = cfg["modules"]["deblur"]
    anomaly_cfg = cfg["modules"]["anomaly_detection"]
    prog_cfg = cfg["progress"]

    folder_name = os.path.basename(folder_path)
    image_paths = scan_images(folder_path, file_cfg["image_extensions"], file_cfg["recursive"])

    if not image_paths:
        logger.warning(f"[{folder_name}] no images found, skipping")
        return {"total": 0, "kept": 0, "rejected": 0}

    logger.info(f"[{folder_name}] found {len(image_paths)} images")

    # Stage 1: Deduplication (batch, 在主线程串行，因需全局比对)
    t_dedup = time.time()
    logger.info(f"[{folder_name}] 开始去重阶段...")
    dedup_mod = DeduplicationModule(dedup_cfg)
    dedup_results = dedup_mod.process_batch(image_paths, logger=logger)
    dedup_map = {r["file_path"]: r for r in dedup_results}
    logger.info(f"[{folder_name}] 去重完成，耗时 {time.time()-t_dedup:.1f}s")

    # Stage 2: Deblur + Anomaly（多线程，ThreadPoolExecutor 避免 Windows spawn 开销）
    num_workers = conc_cfg.get("num_workers") or multiprocessing.cpu_count()
    batch_size = conc_cfg.get("batch_size", 100)
    memory_monitor = MemoryMonitor(conc_cfg.get("memory_limit_mb", 2048))
    all_records: List[Dict[str, Any]] = []
    use_threads = conc_cfg.get("enabled", True) and num_workers > 1

    logger.info(f"[{folder_name}] 开始去模糊+异常检测，workers={num_workers if use_threads else 1}（{'多线程' if use_threads else '单线程'}）")
    t_proc = time.time()

    with ProgressTracker(len(image_paths), f"Processing {folder_name}", prog_cfg.get("enabled", True)) as prog:
        # 线程池在循环外创建，避免每批重建的开销
        executor_ctx = ThreadPoolExecutor(max_workers=num_workers) if use_threads else None
        try:
            for batch_idx, batch_start in enumerate(range(0, len(image_paths), batch_size)):
                batch = image_paths[batch_start: batch_start + batch_size]
                logger.debug(f"[{folder_name}] batch {batch_idx+1}，{len(batch)} 张")

                if use_threads and executor_ctx:
                    futures = {
                        executor_ctx.submit(_process_single_image, p, deblur_cfg, anomaly_cfg): p
                        for p in batch
                    }
                    for future in as_completed(futures):
                        p = futures[future]
                        try:
                            rec = future.result()
                        except Exception as e:
                            logger.error(f"处理失败: {os.path.basename(p)}, 错误: {e}")
                            rec = _build_base_record(p)
                        dedup_rec = dedup_map.get(rec["file_path"], {})
                        rec.update({
                            "phash": dedup_rec.get("phash", ""),
                            "phash_hamming_distance": dedup_rec.get("phash_hamming_distance", -1),
                            "ssim_score": dedup_rec.get("ssim_score", -1.0),
                            "is_duplicate": dedup_rec.get("is_duplicate", "否"),
                            "duplicate_of": dedup_rec.get("duplicate_of", ""),
                        })
                        all_records.append(rec)
                        prog.update(msg=os.path.basename(p))
                else:
                    for p in batch:
                        try:
                            rec = _process_single_image(p, deblur_cfg, anomaly_cfg)
                        except Exception as e:
                            logger.error(f"处理失败: {os.path.basename(p)}, 错误: {e}")
                            rec = _build_base_record(p)
                        dedup_rec = dedup_map.get(p, {})
                        rec.update({
                            "phash": dedup_rec.get("phash", ""),
                            "phash_hamming_distance": dedup_rec.get("phash_hamming_distance", -1),
                            "ssim_score": dedup_rec.get("ssim_score", -1.0),
                            "is_duplicate": dedup_rec.get("is_duplicate", "否"),
                            "duplicate_of": dedup_rec.get("duplicate_of", ""),
                        })
                        all_records.append(rec)
                        prog.update(msg=os.path.basename(p))

                memory_monitor.check_and_collect()
        finally:
            if executor_ctx:
                executor_ctx.shutdown(wait=False)

    logger.info(f"[{folder_name}] 去模糊+异常检测完成，耗时 {time.time()-t_proc:.1f}s")

    # Stage 3: Determine final keep/reject
    for rec in all_records:
        _determine_final_keep(rec)

    # Stage 4: Output
    if not dry_run:
        filtered_dir, csv_path = create_output_dirs(output_base, folder_name)
        for rec in all_records:
            if rec["final_keep"] == "是":
                dst = os.path.join(filtered_dir, rec["filename"])
                shutil.copy2(rec["file_path"], dst)
        ordered_records = [_ensure_ordered_fields(r) for r in all_records]
        write_csv(ordered_records, csv_path, file_cfg["csv_encoding"])
        logger.info(f"[{folder_name}] report written to: {csv_path}")

    kept = sum(1 for r in all_records if r["final_keep"] == "是")
    rejected = len(all_records) - kept
    logger.info(f"[{folder_name}] kept {kept}, rejected {rejected}")
    return {"total": len(all_records), "kept": kept, "rejected": rejected}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.no_dedup:
        cfg["modules"]["deduplication"]["enabled"] = False
    if args.no_deblur:
        cfg["modules"]["deblur"]["enabled"] = False
    if args.no_anomaly:
        cfg["modules"]["anomaly_detection"]["enabled"] = False
    if args.workers is not None:
        cfg["concurrency"]["num_workers"] = args.workers
    if args.verbose:
        cfg["logging"]["level"] = "DEBUG"

    output_base = args.output or cfg["file"]["output_dir"]
    logger = setup_logger("main", config=cfg["logging"])

    if args.dry_run:
        logger.info("=== DRY RUN MODE - files will not be saved ===")

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Scan subfolders
    subfolders = [
        os.path.join(input_dir, d)
        for d in sorted(os.listdir(input_dir))
        if os.path.isdir(os.path.join(input_dir, d)) and d not in ("output", "logs")
    ]

    if not subfolders:
        logger.warning(f"No subfolders in {input_dir}, processing input dir itself")
        subfolders = [input_dir]

    total_stats = {"total": 0, "kept": 0, "rejected": 0}
    for folder in subfolders:
        stats = process_folder(folder, cfg, output_base, args.dry_run, logger)
        for k in total_stats:
            total_stats[k] += stats[k]

    logger.info(
        f"=== Complete === Total {total_stats['total']} images, "
        f"kept {total_stats['kept']}, rejected {total_stats['rejected']}"
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
