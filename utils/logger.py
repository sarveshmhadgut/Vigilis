import os
import logging
import colorlog
from from_root import from_root
from logging.handlers import RotatingFileHandler
from datetime import datetime


def fallback_from_root() -> str:
    return os.getcwd()


def get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# Uncomment the line below if your from_root works; otherwise use fallback
# from_root: Callable[[], str] = fallback_from_root

LOGS_DIR: str = "logs"
LOG_FILE_FORMAT: str = f"{get_current_timestamp()}.log"

maxBytes: int = 5 * 1024 * 1024  # 5 MB
backupCount: int = 4

logs_dirpath: str = os.path.join(from_root(), LOGS_DIR)
os.makedirs(logs_dirpath, exist_ok=True)
log_filepath: str = os.path.join(logs_dirpath, LOG_FILE_FORMAT)


def config_logger() -> None:
    """
    Configures the root logger with a colored console handler and a rotating file handler.

    This function sets up logging to output to both the console (with color-coded levels)
    and a rotating file for persistent storage. Handlers are added only if none exist
    to prevent duplicates. The logger level is set to DEBUG for maximum verbosity.

    Handlers:
    - Console: Uses colorlog for level-specific colors, logs at DEBUG level.
    - File: Rotates files when they exceed maxBytes, keeps up to backupCount backups.

    Raises:
        Any exceptions from handler initialization (e.g., file permission issues).
    """
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Suppress httpx and httpcore INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Suppress FutureWarnings
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    file_format: logging.Formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    console_format: colorlog.ColoredFormatter = colorlog.ColoredFormatter(
        "[ %(asctime)s ] %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    if not logger.handlers:
        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        file_handler: RotatingFileHandler = RotatingFileHandler(
            log_filepath, maxBytes=maxBytes, backupCount=backupCount, encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)


config_logger()
