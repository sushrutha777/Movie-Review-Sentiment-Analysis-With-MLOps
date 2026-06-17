import logging
import os
import random
import sys
from pathlib import Path
import numpy as np

# Base log path
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mlops.log"

def setup_logging(name: str = "mlops") -> logging.Logger:
    """Configures structured logging for console and file."""
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't add more
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)

    # Formatter for structured logs
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
