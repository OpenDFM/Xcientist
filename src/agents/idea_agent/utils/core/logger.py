"""Logging setup helpers for Idea Agent console and file output."""

from contextlib import contextmanager
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Iterator, Optional

from loguru import logger as _loguru_logger


_LOGGER_NAME = "LigAgent"
_MANAGED_LOGGER_PREFIX = "LigAgent"
_DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
_DEFAULT_LOG_DIR = os.path.normpath(_DEFAULT_LOG_DIR)
_DEFAULT_LOG_FILE = "ligagent.log"
_FILE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[logger_name]} | {message}"
_CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{extra[logger_name]}</cyan> | "
    "<level>{message}</level>"
)
_logger: Optional["LoguruCompatLogger"] = None
_loguru_runtime_ready = False
_fallback_sink_id: Optional[int] = None


def _normalize_level(level: int | str) -> str:
    if isinstance(level, str):
        text = level.strip().upper()
        return text or "INFO"
    if level >= logging.CRITICAL:
        return "CRITICAL"
    if level >= logging.ERROR:
        return "ERROR"
    if level >= logging.WARNING:
        return "WARNING"
    if level >= logging.INFO:
        return "INFO"
    if level >= logging.DEBUG:
        return "DEBUG"
    return "TRACE"


def _ensure_loguru_runtime() -> None:
    global _fallback_sink_id, _loguru_runtime_ready
    if _loguru_runtime_ready:
        return
    try:
        _loguru_logger.remove(0)
    except Exception:
        pass
    if _fallback_sink_id is None:
        _fallback_sink_id = _loguru_logger.add(
            sys.stderr,
            level="DEBUG",
            format=_CONSOLE_LOG_FORMAT,
            colorize=True,
            enqueue=True,
            filter=lambda record: not str(record["extra"].get("logger_name", "")).startswith(
                _MANAGED_LOGGER_PREFIX
            ),
        )
    _loguru_runtime_ready = True


class LoguruCompatLogger:
    """A small loguru-backed logger that preserves stdlib `%`-style call sites."""

    def __init__(self, name: str, *, level: int | str = logging.INFO) -> None:
        self.name = name
        self.level = _normalize_level(level)
        self._bound_logger = _loguru_logger.bind(logger_name=name)
        self._desired_file_paths: list[str] = []
        self._include_console = False
        self._file_sink_ids: list[int] = []
        self._console_sink_ids: list[int] = []
        self._console_suspend_depth = 0

    def setLevel(self, level: int | str) -> None:
        normalized = _normalize_level(level)
        if normalized == self.level:
            return
        self.level = normalized
        if self.has_sinks():
            self._rebuild_sinks()

    def has_sinks(self) -> bool:
        return bool(self._desired_file_paths or self._include_console)

    def configure_sinks(
        self,
        *,
        file_paths: list[str] | None = None,
        include_console: bool | None = None,
    ) -> None:
        if file_paths is not None:
            self._desired_file_paths = [str(Path(path)) for path in file_paths]
        if include_console is not None:
            self._include_console = bool(include_console)
        self._rebuild_sinks()

    def clear_sinks(self) -> None:
        self._remove_sink_ids()
        self._desired_file_paths = []
        self._include_console = False

    def _record_matches(self, record: dict) -> bool:
        return record["extra"].get("logger_name") == self.name

    def _console_filter(self, record: dict) -> bool:
        return self._record_matches(record) and self._console_suspend_depth == 0

    def _remove_sink_ids(self) -> None:
        for sink_id in [*self._file_sink_ids, *self._console_sink_ids]:
            try:
                _loguru_logger.remove(sink_id)
            except Exception:
                pass
        self._file_sink_ids.clear()
        self._console_sink_ids.clear()

    def _rebuild_sinks(self) -> None:
        self._remove_sink_ids()
        if not self.has_sinks():
            return
        _ensure_loguru_runtime()
        for path in self._desired_file_paths:
            sink_id = _loguru_logger.add(
                path,
                level=self.level,
                format=_FILE_LOG_FORMAT,
                rotation="5 MB",
                retention=5,
                encoding="utf-8",
                enqueue=True,
                filter=self._record_matches,
            )
            self._file_sink_ids.append(sink_id)
        if self._include_console:
            sink_id = _loguru_logger.add(
                sys.stderr,
                level=self.level,
                format=_CONSOLE_LOG_FORMAT,
                colorize=True,
                enqueue=True,
                filter=self._console_filter,
            )
            self._console_sink_ids.append(sink_id)

    def _format_message(self, message: object, args: tuple[object, ...]) -> str:
        text = str(message)
        if not args:
            return text
        try:
            return text % args
        except Exception:
            return f"{text} | args={args!r}"

    def log(self, level: int | str, message: object, *args: object) -> None:
        if not self.has_sinks():
            return
        _ensure_loguru_runtime()
        self._bound_logger.log(_normalize_level(level), self._format_message(message, args))

    def debug(self, message: object, *args: object) -> None:
        self.log("DEBUG", message, *args)

    def info(self, message: object, *args: object) -> None:
        self.log("INFO", message, *args)

    def warning(self, message: object, *args: object) -> None:
        self.log("WARNING", message, *args)

    def error(self, message: object, *args: object) -> None:
        self.log("ERROR", message, *args)

    def critical(self, message: object, *args: object) -> None:
        self.log("CRITICAL", message, *args)

    def exception(self, message: object, *args: object) -> None:
        if not self.has_sinks():
            return
        _ensure_loguru_runtime()
        self._bound_logger.opt(exception=True).error(self._format_message(message, args))

    @contextmanager
    def suspend_console(self) -> Iterator[None]:
        self._console_suspend_depth += 1
        try:
            yield
        finally:
            self._console_suspend_depth = max(0, self._console_suspend_depth - 1)


def init_logger(
    log_dir: Optional[str] = None,
    filename: str = _DEFAULT_LOG_FILE,
    level: int = logging.DEBUG,
    include_console: bool = True,
    include_timestamp: bool = True,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    force_reinit: bool = False,
) -> LoguruCompatLogger:
    """
    Initialize or re-configure the process-wide logger.
    - Rotating file sink (5MB, 5 backups)
    - Optional console sink
    - Optional timestamp appended to filename (before file extension)
    - Set force_reinit=True to rebuild sinks even if the logger already exists.
    """
    global _logger
    logger = _logger or LoguruCompatLogger(_LOGGER_NAME, level=level)
    logger.setLevel(level)

    if force_reinit or not logger.has_sinks():
        resolved_log_dir = log_dir or _DEFAULT_LOG_DIR
        os.makedirs(resolved_log_dir, exist_ok=True)

        resolved_filename = filename
        if include_timestamp:
            name, ext = os.path.splitext(filename)
            if not ext:
                ext = ".log"
            ts = datetime.now().strftime(timestamp_format)
            resolved_filename = f"{name}_{ts}{ext}"

        file_path = os.path.join(resolved_log_dir, resolved_filename)
        logger.configure_sinks(file_paths=[file_path], include_console=include_console)

    _logger = logger
    return logger


def get_logger() -> LoguruCompatLogger:
    global _logger
    if _logger is None:
        _logger = LoguruCompatLogger(_LOGGER_NAME, level=logging.INFO)
    return _logger


def normalize_mode_log_key(idea_taste_mode: Optional[str]) -> str:
    return str(idea_taste_mode or "default").strip() or "default"


def mode_log_filename(idea_taste_mode: Optional[str]) -> str:
    raw_mode = normalize_mode_log_key(idea_taste_mode).lower()
    safe_mode = "".join(ch if ch.isalnum() else "_" for ch in raw_mode).strip("_")
    return f"ligagent_pro_{safe_mode or 'default'}.log"


def get_or_create_mode_logger(
    mode_loggers: Dict[str, LoguruCompatLogger],
    base_logger: LoguruCompatLogger,
    run_dir: Path,
    idea_taste_mode: Optional[str],
) -> LoguruCompatLogger:
    mode_key = normalize_mode_log_key(idea_taste_mode)
    existing = mode_loggers.get(mode_key)
    if existing is not None:
        return existing

    mode_logger = LoguruCompatLogger(f"LigAgent.{mode_key}", level=base_logger.level)
    log_path = run_dir / "logs" / mode_log_filename(idea_taste_mode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode_logger.configure_sinks(file_paths=[str(log_path)], include_console=False)
    mode_loggers[mode_key] = mode_logger
    return mode_logger


@contextmanager
def suspend_console_handlers(logger: LoguruCompatLogger) -> Iterator[None]:
    with logger.suspend_console():
        yield
