"""
src/evaluate.py — Model evaluation and visualisation utilities.

Provides functions for computing metrics, plotting confusion matrices,
ROC curves, and comparing multiple models side-by-side.
"""

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FIGURES_DIR, get_logger

logger = get_logger(__name__)

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    model_name: str = "Model",
    y_scores: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute standard classification metrics and print a report.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        model_name: Display name used in log messages (default ``'Model'``).
        y_scores: Probability scores for the positive class (shape ``(n,)``).
                  Required for ROC-AUC. If *None*, AUC is set to ``NaN``.

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``roc_auc``.

    Example:
        >>> results = evaluate_model(y_true, y_pred, "Baseline")
        >>> results['f1']
        0.891
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    if y_scores is not None:
        try:
            # Works for binary and multi-class (OvR)
            roc_auc = roc_auc_score(y_true, y_scores, multi_class="ovr")
        except ValueError:
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")

    logger.info(
        "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        model_name, acc, prec, rec, f1, roc_auc,
    )
    logger.info("\n%s", classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    model_name: str = "Model",
    labels: List[str] | None = None,
) -> Path:
    """Plot and save a normalised confusion matrix heat-map.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        model_name: Used in the plot title and output filename.
        labels: Optional list of class name strings for axis tick labels.

    Returns:
        Path to the saved PNG file.

    Example:
        >>> path = plot_confusion_matrix(y_true, y_pred, "BERT")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if labels is None:
        labels = [str(c) for c in unique_classes]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    # Sanitise model name for filename
    safe_name = model_name.lower().replace(" ", "_")
    save_path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    baseline_results: Dict[str, float],
    bert_results: Dict[str, float],
) -> pd.DataFrame:
    """Build a comparison DataFrame and save a bar-chart figure.

    Args:
        baseline_results: Metrics dict returned by :func:`evaluate_model`
                          for the baseline model.
        bert_results: Metrics dict for the BERT model.

    Returns:
        DataFrame with models as rows and metrics as columns.

    Side-effects:
        Saves ``reports/figures/model_comparison.png``.

    Example:
        >>> compare_df = compare_models(baseline_results, bert_results)
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    df = pd.DataFrame(
        {
            "Baseline (TF-IDF + LR)": [baseline_results.get(m, 0) for m in metrics],
            "BERT Fine-tuned": [bert_results.get(m, 0) for m in metrics],
        },
        index=[m.capitalize() for m in metrics],
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, df["Baseline (TF-IDF + LR)"], width, label="Baseline", color="#4C72B0")
    ax.bar(x + width / 2, df["BERT Fine-tuned"], width, label="BERT", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    plt.tight_layout()

    save_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", save_path)

    return df.T   # models as rows


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: List[int] | np.ndarray,
    y_scores: np.ndarray,
    model_name: str = "Model",
) -> Path:
    """Plot and save the ROC curve with AUC annotation.

    For binary classification ``y_scores`` should be the probability of the
    positive class (shape ``(n,)``).

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_scores: Positive-class probability scores.
        model_name: Used in the plot title and filename.

    Returns:
        Path to the saved PNG file.

    Example:
        >>> path = plot_roc_curve(y_test, probs[:, 1], "Baseline")
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    save_path = FIGURES_DIR / f"roc_curve_{safe_name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved to %s", save_path)
    return save_path
