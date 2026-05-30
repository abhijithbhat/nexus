"""Structured rotating logger factory. Usage: logger = get_logger(__name__)"""
import logging
import sys
from logging.handlers import RotatingFileHandler


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Idempotent — safe to call multiple times."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console: INFO and above
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    # File: DEBUG and above, rotating at 10 MB, keep 3 backups
    try:
        fh = RotatingFileHandler("nexus.log", maxBytes=10_000_000, backupCount=3)
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except OSError:
        pass  # Not writable — console only

    return logger
