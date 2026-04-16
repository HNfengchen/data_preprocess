import os
import csv
import pytest
from tests.conftest import make_image
from utils.file_handler import scan_images, validate_image, create_output_dirs, write_csv


class TestScanImages:
    def test_finds_jpg_and_png(self, tmp_image_dir):
        make_image(os.path.join(tmp_image_dir, "a.jpg"))
        make_image(os.path.join(tmp_image_dir, "b.png"))
        open(os.path.join(tmp_image_dir, "c.txt"), "w").close()
        result = scan_images(tmp_image_dir, [".jpg", ".png"])
        basenames = {os.path.basename(p) for p in result}
        assert basenames == {"a.jpg", "b.png"}

    def test_recursive_false_skips_subdirs(self, tmp_image_dir):
        sub = os.path.join(tmp_image_dir, "sub")
        os.makedirs(sub)
        make_image(os.path.join(sub, "nested.jpg"))
        make_image(os.path.join(tmp_image_dir, "root.jpg"))
        result = scan_images(tmp_image_dir, [".jpg"], recursive=False)
        basenames = {os.path.basename(p) for p in result}
        assert basenames == {"root.jpg"}

    def test_recursive_true_includes_subdirs(self, tmp_image_dir):
        sub = os.path.join(tmp_image_dir, "sub")
        os.makedirs(sub)
        make_image(os.path.join(sub, "nested.jpg"))
        make_image(os.path.join(tmp_image_dir, "root.jpg"))
        result = scan_images(tmp_image_dir, [".jpg"], recursive=True)
        assert len(result) == 2

    def test_empty_dir_returns_empty(self, tmp_image_dir):
        assert scan_images(tmp_image_dir, [".jpg"]) == []


class TestValidateImage:
    def test_valid_image_returns_true(self, sample_normal_image):
        valid, err = validate_image(sample_normal_image)
        assert valid is True
        assert err == ""

    def test_corrupt_file_returns_false(self, tmp_image_dir):
        p = os.path.join(tmp_image_dir, "bad.jpg")
        with open(p, "wb") as f:
            f.write(b"not an image at all")
        valid, err = validate_image(p)
        assert valid is False
        assert err != ""


class TestWriteCsv:
    def test_writes_rows_with_header(self, tmp_image_dir):
        out = os.path.join(tmp_image_dir, "report.csv")
        rows = [{"filename": "a.jpg", "final_keep": "是"}, {"filename": "b.jpg", "final_keep": "否"}]
        write_csv(rows, out, encoding="utf-8-sig")
        with open(out, encoding="utf-8-sig") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["filename"] == "a.jpg"
