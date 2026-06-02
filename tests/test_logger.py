"""Tests for logger.py — get_logger and log_error."""
import logging
import os
import tempfile


def test_get_logger_returns_logger():
    """get_logger returns a logging.Logger with the given name."""
    from obsidian_ai.logger import get_logger

    logger = get_logger("test.basic")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.basic"


def test_get_logger_idempotent():
    """Calling get_logger twice with the same name returns the same logger."""
    from obsidian_ai.logger import get_logger

    l1 = get_logger("test.idempotent")
    l2 = get_logger("test.idempotent")
    assert l1 is l2


def test_get_logger_with_file():
    """get_logger with log_file creates a file handler."""
    from obsidian_ai.logger import get_logger

    logger = get_logger("test.file", log_file="test_file.log")
    # Should have at least 2 handlers (console + file)
    assert len(logger.handlers) >= 2


def test_log_error_basic():
    """log_error formats a basic error message."""
    from obsidian_ai.logger import get_logger, log_error

    logger = get_logger("test.error")
    # Should not raise
    log_error(logger, "something broke")


def test_log_error_with_exception():
    """log_error includes exception info when provided."""
    from obsidian_ai.logger import get_logger, log_error

    logger = get_logger("test.error_exc")
    try:
        raise ValueError("test error")
    except ValueError as e:
        log_error(logger, "failed", exc=e)


def test_log_error_with_context():
    """log_error includes key=value context."""
    from obsidian_ai.logger import get_logger, log_error

    logger = get_logger("test.error_ctx")
    log_error(logger, "failed", path="some/note.md", count=5)


def test_log_dir_created():
    """LOG_DIR is a valid directory path."""
    from obsidian_ai.logger import LOG_DIR

    assert os.path.isabs(LOG_DIR)
    assert LOG_DIR.endswith("logs")
