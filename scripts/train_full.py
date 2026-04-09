"""
scripts/train_full.py — End-to-end training on real TweetEval data.

Trains both models and saves all metrics + figures:
  1. Baseline  — TF-IDF + Logistic Regression
  2. BERT      — Fine-tuned bert-base-uncased on GPU

Usage:
    python scripts/train_full.py                  # both models
    python scripts/train_full.py --model baseline # baseline only
    python scripts/train_full.py --model bert     # BERT only

Requirements:
    - data/raw/tweet_eval_sentiment.csv  (run scripts/download_data.py first)
    - GPU strongly recommended for BERT
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    TWEET_EVAL_PATH,
    RANDOM_SEED,
    get_logger,
    set_seed,
)

logger = get_logger(__name__)
set_seed(RANDOM_SEED)

METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "metrics.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits():
    """Load tweet_eval CSV and return pre-defined train/val/test splits."""
    from src.data_loader import preprocess_dataframe

    logger.info("Loading data from %s", TWEET_EVAL_PATH)
    df = pd.read_csv(TWEET_EVAL_PATH)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    train_raw = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].copy().reset_index(drop=True)
    test_raw = df[df["split"] == "test"].copy().reset_index(drop=True)

    logger.info(
        "Sizes — train: %d | val: %d | test: %d",
        len(train_raw), len(val_raw), len(test_raw),
    )

    train_df = preprocess_dataframe(train_raw)
    val_df = preprocess_dataframe(val_raw)
    test_df = preprocess_dataframe(test_raw)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------

def save_metrics(key: str, results: dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    existing[key] = results
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info("Metrics saved → %s", METRICS_PATH)


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def run_baseline(train_df, val_df, test_df) -> dict:
    from src.baseline_model import train_baseline, predict
    from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

    logger.info("=" * 60)
    logger.info("BASELINE: TF-IDF + Logistic Regression")
    logger.info("=" * 60)

    t0 = time.time()
    pipeline = train_baseline(train_df, val_df)
    logger.info("Training took %.1f s", time.time() - t0)

    labels, probs = predict(pipeline, test_df["clean_text"].tolist())
    y_true = test_df["label"].values

    results = evaluate_model(y_true, labels, "Baseline (TF-IDF + LR)", y_scores=probs)
    plot_confusion_matrix(y_true, labels, "Baseline (TF-IDF + LR)")
    plot_roc_curve(y_true, probs, "Baseline (TF-IDF + LR)")
    save_metrics("baseline", results)
    return results


# ---------------------------------------------------------------------------
# BERT training
# ---------------------------------------------------------------------------

def run_bert(train_df, val_df, test_df) -> dict:
    import torch
    import numpy as np
    from src.bert_model import train_bert, predict_bert
    from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 60)
    logger.info("BERT fine-tuning | device=%s | train=%d val=%d test=%d",
                device, len(train_df), len(val_df), len(test_df))
    logger.info("=" * 60)

    if device == "cpu":
        logger.warning("No GPU — BERT training on CPU will be very slow.")

    t0 = time.time()
    model, tokenizer = train_bert(train_df, val_df)
    logger.info("Training took %.1f min", (time.time() - t0) / 60)

    labels, confidences = predict_bert(model, tokenizer, test_df["clean_text"].tolist())
    y_true = test_df["label"].values
    num_classes = len(set(y_true))

    # For evaluate_model we need a (n, num_classes) probability matrix
    # predict_bert returns confidence for the predicted class only
    # Build a proxy matrix: predicted class gets confidence, others share the rest
    probs_matrix = np.zeros((len(labels), num_classes))
    for i, (lbl, conf) in enumerate(zip(labels, confidences)):
        probs_matrix[i, lbl] = conf
        remainder = (1.0 - conf) / max(num_classes - 1, 1)
        for c in range(num_classes):
            if c != lbl:
                probs_matrix[i, c] = remainder

    results = evaluate_model(y_true, labels, "BERT Fine-tuned", y_scores=probs_matrix)
    plot_confusion_matrix(y_true, labels, "BERT Fine-tuned")
    plot_roc_curve(y_true, probs_matrix, "BERT Fine-tuned")
    save_metrics("bert", results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Social Sentiment Tracker models")
    parser.add_argument(
        "--model", choices=["baseline", "bert", "both"], default="both",
        help="Which model to train (default: both)",
    )
    args = parser.parse_args()

    if not TWEET_EVAL_PATH.exists():
        logger.error(
            "Dataset not found at %s\nRun first:  python scripts/download_data.py",
            TWEET_EVAL_PATH,
        )
        sys.exit(1)

    train_df, val_df, test_df = load_splits()

    if args.model in ("baseline", "both"):
        baseline_results = run_baseline(train_df, val_df, test_df)
        print(f"\nBaseline results: {baseline_results}\n")

    if args.model in ("bert", "both"):
        bert_results = run_bert(train_df, val_df, test_df)
        print(f"\nBERT results: {bert_results}\n")

    if args.model == "both":
        with open(METRICS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        if "baseline" in saved and "bert" in saved:
            from src.evaluate import compare_models
            compare_models(saved["baseline"], saved["bert"])

    print(f"\nDone. Metrics → {METRICS_PATH}")
