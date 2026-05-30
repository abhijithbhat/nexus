import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from utils.config import settings, ROOT_DIR

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't duplicate them
    if logger.handlers:
        return logger
        
    log_level_str = settings.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )
    
    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    
    # Rotating File Handler
    log_file_path = ROOT_DIR / "nexus.log"
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    
    # Prevent propagation to the root logger to avoid duplicate output in some environments
    logger.propagate = False
    
    return logger
