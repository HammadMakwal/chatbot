# logger_setup.py
import logging
import os
from datetime import datetime

def get_logger(name="chatbot", logs_dir="logs", level=logging.INFO):
    """
    Returns a logger that logs both to console and to a session file in logs_dir.
    Each session file is named session_YYYY-MM-DD_HH-MM-SS.log
    """
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = os.path.join(logs_dir, f"session_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers repeatedly if function is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # attach logfile path for convenience
    logger.logfile = logfile
    return logger
