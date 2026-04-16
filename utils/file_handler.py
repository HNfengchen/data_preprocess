import os
import csv
from PIL import Image
from typing import List, Tuple


def scan_images(folder: str, extensions: List[str], recursive: bool = False) -> List[str]:
    """Scan folder and return all files matching extensions (sorted)"""
    exts = {e.lower() for e in extensions}
    result = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    result.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in exts:
                full = os.path.join(folder, f)
                if os.path.isfile(full):
                    result.append(full)
    return sorted(result)


def validate_image(file_path: str) -> Tuple[bool, str]:
    """
    Validate image can be read and is complete.
    Returns (True, "") or (False, error_message).
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
        return True, ""
    except Exception as e:
        return False, str(e)


def create_output_dirs(base_output: str, folder_name: str) -> Tuple[str, str]:
    """
    Create filtered_images/ directory under base_output/folder_name/.
    Returns (filtered_images_path, csv_path).
    """
    filtered_dir = os.path.join(base_output, folder_name, "filtered_images")
    os.makedirs(filtered_dir, exist_ok=True)
    csv_path = os.path.join(base_output, folder_name, "analysis_report.csv")
    return filtered_dir, csv_path


def write_csv(rows: List[dict], output_path: str, encoding: str = "utf-8-sig") -> None:
    """Write rows (list of dicts) to CSV. Infer column names from first row."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
