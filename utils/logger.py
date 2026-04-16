import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_dir: str = "logs", config: dict | None = None) -> logging.Logger:
    """
    Create or retrieve a named logger.
    Outputs to both console (INFO+) and rotating file (DEBUG+).
    """
    if config is None:
        config = {}

    level_str = config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    max_bytes = config.get("max_bytes", 10 * 1024 * 1024)
    backup_count = config.get("backup_count", 3)
    fmt = config.get("format", "[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
    datefmt = config.get("date_format", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already initialized

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
