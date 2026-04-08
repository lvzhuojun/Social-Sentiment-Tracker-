"""
src/baseline_model.py — TF-IDF + Logistic Regression baseline pipeline.

Provides a fast, interpretable sentiment classifier that can be trained in
seconds on CPU and serves as the performance benchmark for the BERT model.
"""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    BASELINE_MODEL_PATH,
    LR_C,
    LR_MAX_ITER,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Construct the TF-IDF → Logistic Regression sklearn Pipeline.

    Returns:
        Untrained :class:`sklearn.pipeline.Pipeline` with steps:
        * ``tfidf`` — :class:`TfidfVectorizer`
        * ``clf``   — :class:`LogisticRegression`

    Example:
        >>> pipe = build_pipeline()
        >>> pipe.fit(X_train, y_train)
    """
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,          # replace tf with 1 + log(tf) to dampen frequency
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",    # ignore single-character tokens
    )
    clf = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_SEED,
        solver="lbfgs",
        multi_class="auto",
    )
    return Pipeline(steps=[("tfidf", tfidf), ("clf", clf)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
) -> Pipeline:
    """Train the TF-IDF + LR pipeline and evaluate on the validation set.

    Args:
        train_df: Training DataFrame with *text_col* and *label_col* columns.
        val_df: Validation DataFrame with the same columns.
        text_col: Name of the text column (default ``'clean_text'``).
        label_col: Name of the label column (default ``'label'``).

    Returns:
        Trained :class:`sklearn.pipeline.Pipeline`.

    Side-effects:
        * Prints validation metrics to stdout via the logger.
        * Saves the pipeline to ``config.BASELINE_MODEL_PATH``.

    Example:
        >>> pipeline = train_baseline(train_df, val_df)
    """
    logger.info("Training baseline TF-IDF + LR …")
    pipeline = build_pipeline()

    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()
    X_val = val_df[text_col].tolist()
    y_val = val_df[label_col].tolist()

    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # Evaluation on validation set
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    logger.info("Validation results — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
                acc, prec, rec, f1)
    logger.info("\n%s", classification_report(y_val, y_pred, zero_division=0))

    # Persist model
    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BASELINE_MODEL_PATH)
    logger.info("Model saved to %s", BASELINE_MODEL_PATH)

    return pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    pipeline: Pipeline,
    texts: list[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with a trained pipeline.

    Args:
        pipeline: Trained sklearn Pipeline (from :func:`train_baseline`
                  or loaded via ``joblib.load``).
        texts: List of raw or pre-cleaned text strings.

    Returns:
        Tuple ``(labels, probabilities)`` where:
        * ``labels`` — 1-D int array of predicted class indices.
        * ``probabilities`` — 2-D float array of shape ``(n_samples, n_classes)``.

    Example:
        >>> labels, probs = predict(pipeline, ["I love this!", "This is terrible."])
        >>> labels
        array([1, 0])
    """
    labels: np.ndarray = pipeline.predict(texts)
    probabilities: np.ndarray = pipeline.predict_proba(texts)
    return labels, probabilities


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_baseline_model(path: Path | None = None) -> Pipeline:
    """Load a persisted baseline pipeline from disk.

    Args:
        path: Path to the ``.pkl`` file. Defaults to ``config.BASELINE_MODEL_PATH``.

    Returns:
        Loaded :class:`sklearn.pipeline.Pipeline`.

    Raises:
        FileNotFoundError: If the model file does not exist.

    Example:
        >>> pipeline = load_baseline_model()
    """
    model_path = Path(path) if path else BASELINE_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {model_path}. "
            "Run train_baseline() first."
        )
    logger.info("Loading baseline model from %s", model_path)
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_data, split_data

    df = load_data()
    train_df, val_df, test_df = split_data(df)
    pipeline = train_baseline(train_df, val_df)

    sample_texts = ["I love this amazing product!", "This is absolutely terrible."]
    preds, probs = predict(pipeline, sample_texts)
    for text, label, prob in zip(sample_texts, preds, probs):
        sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
        conf = prob.max()
        print(f"  [{sentiment} {conf:.2%}] {text}")
