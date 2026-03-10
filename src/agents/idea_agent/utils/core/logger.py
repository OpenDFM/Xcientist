"""Logging setup helpers for Idea Agent console and file output."""

from contextlib import contextmanager
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterator, Optional
from datetime import datetime


_LOGGER_NAME = "LigAgent"
_DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
_DEFAULT_LOG_DIR = os.path.normpath(_DEFAULT_LOG_DIR)
_DEFAULT_LOG_FILE = "ligagent.log"
_logger: Optional[logging.Logger] = None


def init_logger(
    log_dir: Optional[str] = None,
    filename: str = _DEFAULT_LOG_FILE,
    level: int = logging.DEBUG,
    include_console: bool = True,
    include_timestamp: bool = True,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    force_reinit: bool = False,
) -> logging.Logger:
    """
    Initialize or re-configure the process-wide logger.
    - Rotating file handler (5MB, 5 backups)
    - Optional console handler
    - Optional timestamp appended to filename (before file extension)
    - Set force_reinit=True to rebuild handlers even if the logger already exists.
    """
    global _logger
    logger = _logger or logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if force_reinit or not logger.handlers:
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)

        resolved_log_dir = log_dir or _DEFAULT_LOG_DIR
        os.makedirs(resolved_log_dir, exist_ok=True)

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
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        _logger = logger
    return _logger


def normalize_mode_log_key(idea_taste_mode: Optional[str]) -> str:
    return str(idea_taste_mode or "default").strip() or "default"


def mode_log_filename(idea_taste_mode: Optional[str]) -> str:
    raw_mode = normalize_mode_log_key(idea_taste_mode).lower()
    safe_mode = "".join(ch if ch.isalnum() else "_" for ch in raw_mode).strip("_")
    return f"ligagent_pro_{safe_mode or 'default'}.log"


def get_or_create_mode_logger(
    mode_loggers: Dict[str, logging.Logger],
    base_logger: logging.Logger,
    run_dir: Path,
    idea_taste_mode: Optional[str],
) -> logging.Logger:
    mode_key = normalize_mode_log_key(idea_taste_mode)
    existing = mode_loggers.get(mode_key)
    if existing is not None:
        return existing

    mode_logger = logging.getLogger(f"LigAgent.{mode_key}")
    mode_logger.setLevel(base_logger.level)
    mode_logger.propagate = False

    for handler in list(mode_logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        mode_logger.removeHandler(handler)

    log_path = run_dir / "logs" / mode_log_filename(idea_taste_mode)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s |  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(mode_logger.level)
    file_handler.setFormatter(formatter)
    mode_logger.addHandler(file_handler)

    mode_loggers[mode_key] = mode_logger
    return mode_logger


@contextmanager
def suspend_console_handlers(logger: logging.Logger) -> Iterator[None]:
    removed_handlers = []
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)
            removed_handlers.append(handler)
    try:
        yield
    finally:
        for handler in removed_handlers:
            logger.addHandler(handler)
