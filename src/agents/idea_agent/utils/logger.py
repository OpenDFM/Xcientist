import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


_LOGGER_NAME = "LigAgent"
_DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
_DEFAULT_LOG_DIR = os.path.normpath(_DEFAULT_LOG_DIR)
_DEFAULT_LOG_FILE = "ligagent.log"
_logger: Optional[logging.Logger] = None


def init_logger(
    log_dir: Optional[str] = None,
    filename: str = _DEFAULT_LOG_FILE,
    level: int = logging.INFO,
    include_console: bool = True,
    include_timestamp: bool = True,
    timestamp_format: str = "%Y%m%d_%H%M%S"
) -> logging.Logger:
    """
    Initialize a process-wide global logger. Safe to call multiple times.
    - Rotating file handler (5MB, 5 backups)
    - Optional console handler
    - Optional timestamp appended to filename (before file extension)
    """
    global _logger
    if _logger is not None:
        _logger.setLevel(level)
        return _logger

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        resolved_log_dir = log_dir or _DEFAULT_LOG_DIR
        os.makedirs(resolved_log_dir, exist_ok=True)

        # append timestamp before extension if requested
        if include_timestamp:
            name, ext = os.path.splitext(filename)
            if not ext:
                ext = ".log"
            ts = datetime.now().strftime(timestamp_format)
            filename = f"{name}_{ts}{ext}"

        file_path = os.path.join(resolved_log_dir, filename)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s |  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(file_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if include_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        return init_logger()
    return _logger
