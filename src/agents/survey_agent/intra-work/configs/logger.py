import logging
import sys
from typing import Optional
from pathlib import Path

from configs.config import BASE_DIR


class ColorFormatter(logging.Formatter):
    """Colored log formatter"""

    COLOR_CODES = {
        logging.ERROR: "\033[91m",  # Red
        logging.WARNING: "\033[93m",  # Yellow
        logging.INFO: "\033[92m",  # Green
        logging.DEBUG: "\033[37m",  # Gray
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET_CODE}"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColorFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # File handler with default formatting
    output_dir = Path(f"{BASE_DIR}/outputs/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / f"{name}.log", "a")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    # Example usage
    logger = get_logger("my_logger")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
