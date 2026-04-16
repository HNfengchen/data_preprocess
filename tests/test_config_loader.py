import pytest
import os
import yaml
import tempfile
from utils.config_loader import load_config, DEFAULT_CONFIG


def write_yaml(d: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(d, f)


class TestLoadConfig:
    def test_returns_default_when_file_missing(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg["modules"]["deduplication"]["enabled"] is True
        assert cfg["concurrency"]["batch_size"] == 100

    def test_merges_user_values_over_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        write_yaml({"modules": {"deblur": {"bren_threshold": 99}}}, str(p))
        cfg = load_config(str(p))
        assert cfg["modules"]["deblur"]["bren_threshold"] == 99
        # 其他默认值保留
        assert cfg["modules"]["deduplication"]["enabled"] is True

    def test_raises_on_invalid_ssim_win_size(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        write_yaml({"modules": {"deduplication": {"ssim_win_size": 4}}}, str(p))
        with pytest.raises(ValueError, match="ssim_win_size"):
            load_config(str(p))

    def test_raises_on_negative_workers(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        write_yaml({"concurrency": {"num_workers": -1}}, str(p))
        with pytest.raises(ValueError, match="num_workers"):
            load_config(str(p))

    def test_raises_on_invalid_batch_size(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        write_yaml({"concurrency": {"batch_size": 0}}, str(p))
        with pytest.raises(ValueError, match="batch_size"):
            load_config(str(p))

    def test_load_none_returns_default(self):
        cfg = load_config(None)
        assert cfg["concurrency"]["batch_size"] == 100
        assert cfg["modules"]["deduplication"]["enabled"] is True
