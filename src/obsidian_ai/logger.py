import logging
import logging.handlers
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file:
            log_path = os.path.join(LOG_DIR, log_file)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def log_error(logger: logging.Logger, msg: str, exc: Exception | None = None, **context):
    parts = [msg]
    if context:
        ctx_str = " — ".join(f"{k}={v}" for k, v in context.items())
        parts.append(f"Context: {ctx_str}")
    full_msg = "\n  ".join(parts)

    if exc:
        tb = traceback.format_exception(exc)
        tb_str = "".join(tb).strip()
        full_msg += f"\n  Traceback: {tb_str}"

    logger.error(full_msg)


def log_search(logger: logging.Logger, query: str, results_count: int,
               signals: list[str], duration_ms: float):
    """Structured log line for search operations."""
    sig_str = ",".join(sorted(set(signals))) if signals else "(none)"
    logger.info("search | query=%s | results=%s | signals=%s | %.1fms",
                query[:80], results_count, sig_str, duration_ms)
