"""
scripts/download_data.py — Download TweetEval sentiment dataset from HuggingFace.

Saves data as data/raw/tweet_eval_sentiment.csv with project label convention:
  0 = negative  |  1 = positive  |  2 = neutral

(TweetEval original: 0=negative, 1=neutral, 2=positive — remapped here.)

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TWEET_EVAL_PATH, RAW_DATA_DIR, get_logger

logger = get_logger(__name__)

HF_DATASET = "tweet_eval"
HF_SUBSET = "sentiment"

# Remap tweet_eval labels to project convention: 0=neg, 1=pos, 2=neutral
_LABEL_REMAP = {0: 0, 1: 2, 2: 1}


def download_tweet_eval() -> None:
    """Download TweetEval sentiment splits and save as a single CSV."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Run: pip install datasets")
        sys.exit(1)

    import pandas as pd

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s/%s from HuggingFace…", HF_DATASET, HF_SUBSET)
    ds = load_dataset(HF_DATASET, HF_SUBSET)

    frames = []
    for split_name in ("train", "validation", "test"):
        split = ds[split_name]
        df = pd.DataFrame({
            "id": range(len(split)),
            "label": [_LABEL_REMAP[lbl] for lbl in split["label"]],
            "text": split["text"],
            "split": split_name,
        })
        frames.append(df)
        logger.info("  %s: %d rows", split_name, len(df))

    full = pd.concat(frames, ignore_index=True)

    full.to_csv(TWEET_EVAL_PATH, index=False)
    logger.info(
        "Saved %d total rows → %s\nLabel distribution: %s",
        len(full),
        TWEET_EVAL_PATH,
        full["label"].value_counts().sort_index().to_dict(),
    )
    print("\nLabel mapping: 0=Negative  1=Positive  2=Neutral")
    print(f"Saved to: {TWEET_EVAL_PATH}")


if __name__ == "__main__":
    download_tweet_eval()
