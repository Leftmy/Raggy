import logging
import sys


def setup_logger(name: str = "rag_app", level: str = "INFO") -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
