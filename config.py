"""
config.py — Global configuration for Social Sentiment Tracker.

All paths, hyperparameters, and logging settings are centralised here
so every other module imports from a single source of truth.
"""

import logging
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root (this file lives at the repo root)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Primary dataset file (Sentiment140 or mock fallback)
SENTIMENT140_PATH: Path = RAW_DATA_DIR / "twitter_training.csv"
# TweetEval sentiment dataset (preferred when available)
TWEET_EVAL_PATH: Path = RAW_DATA_DIR / "tweet_eval_sentiment.csv"
MOCK_DATA_PATH: Path = RAW_DATA_DIR / "mock_data.csv"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS_DIR: Path = ROOT_DIR / "models"
BASELINE_MODEL_PATH: Path = MODELS_DIR / "baseline_tfidf_lr.pkl"
BERT_MODEL_PATH: Path = MODELS_DIR / "bert_sentiment.pt"

# ---------------------------------------------------------------------------
# Reports / figures
# ---------------------------------------------------------------------------
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# BERT / HuggingFace settings
# ---------------------------------------------------------------------------
BERT_MODEL_NAME: str = "bert-base-uncased"
MAX_LENGTH: int = 128

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 16
EPOCHS: int = 3
LEARNING_RATE: float = 2e-5
WARMUP_RATIO: float = 0.1          # fraction of total steps used for warmup

# TF-IDF baseline
TFIDF_MAX_FEATURES: int = 50_000
TFIDF_NGRAM_RANGE: tuple = (1, 2)
LR_C: float = 1.0
LR_MAX_ITER: int = 1000

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility across random / numpy / torch.

    Args:
        seed: Integer seed value. Defaults to ``RANDOM_SEED``.

    Example:
        >>> from config import set_seed
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not installed — skip


# ---------------------------------------------------------------------------
# Data split ratios
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.1   # fraction of the *remaining* train split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: int = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example:
        >>> from config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Logger ready.")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


# ---------------------------------------------------------------------------
# Ensure critical directories exist at import time
# ---------------------------------------------------------------------------
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
