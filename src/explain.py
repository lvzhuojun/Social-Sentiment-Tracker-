"""
src/explain.py — SHAP-based prediction explanation for the baseline model.

Provides token-level attribution showing which words pushed the model
toward or away from the predicted sentiment class.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_logger

logger = get_logger(__name__)


def explain_baseline_prediction(
    pipeline,
    text: str,
    n_top: int = 12,
) -> Tuple[List[Tuple[str, float]], int, List[int]]:
    """Compute SHAP token attributions for a single baseline-model prediction.

    Uses ``shap.LinearExplainer`` on the Logistic Regression step of the
    TF-IDF pipeline.  A zero-vector background is used (equivalent to
    "no words present"), so each SHAP value represents the contribution of
    that word relative to an empty document.

    Args:
        pipeline: Trained sklearn Pipeline with steps ``tfidf`` and ``clf``.
        text: Pre-cleaned input string to explain.
        n_top: Maximum number of top-contributing tokens to return.

    Returns:
        Tuple ``(contributions, predicted_class, classes)`` where:

        * ``contributions`` — list of ``(token, shap_value)`` sorted by
          ``abs(shap_value)`` descending.  Positive values push toward the
          predicted class; negative values push away.
        * ``predicted_class`` — integer label predicted by the pipeline.
        * ``classes`` — list of integer class labels known to the classifier.

    Raises:
        ImportError: If ``shap`` is not installed.

    Example:
        >>> contribs, pred, classes = explain_baseline_prediction(pipeline, "great product")
        >>> contribs[0]
        ('great', 0.342)
    """
    try:
        import shap
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "shap is required for explanations. "
            "Install it with: pip install shap"
        ) from exc

    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Transform input text to TF-IDF feature vector
    X = vectorizer.transform([text])  # shape: (1, n_features)

    # Zero-vector background = "empty document" baseline
    background = sp.csr_matrix(X.shape)

    explainer = shap.LinearExplainer(clf, background)
    shap_values = explainer.shap_values(X)
    # shap_values: list of (1, n_features) arrays for each class (multi-class),
    # or a single (1, n_features) array (binary)

    # Identify predicted class and its index in clf.classes_
    pred_class = int(clf.predict(X)[0])
    classes = clf.classes_.tolist()
    class_idx = classes.index(pred_class)

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[class_idx]).flatten()
    else:
        sv = np.asarray(shap_values).flatten()

    # Only consider features (words) that are non-zero in the input text
    nonzero_cols = X.nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()

    contributions = [
        (str(feature_names[i]), float(sv[i]))
        for i in nonzero_cols
    ]
    # Sort by absolute SHAP value — most influential first
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    logger.info(
        "SHAP explanation: pred=%d, top token='%s' (%.4f)",
        pred_class,
        contributions[0][0] if contributions else "—",
        contributions[0][1] if contributions else 0.0,
    )

    return contributions[:n_top], pred_class, classes


def shap_to_plotly_bar(
    contributions: List[Tuple[str, float]],
    predicted_class: int,
    label_names: dict | None = None,
):
    """Render SHAP token attributions as a horizontal Plotly bar chart.

    Args:
        contributions: List of ``(token, shap_value)`` tuples (from
            :func:`explain_baseline_prediction`).
        predicted_class: Predicted integer label.
        label_names: Optional mapping from label int to display string.
            Defaults to ``{0: 'Negative', 1: 'Positive', 2: 'Neutral'}``.

    Returns:
        :class:`plotly.graph_objects.Figure`.

    Example:
        >>> fig = shap_to_plotly_bar(contributions, pred_class=1)
        >>> fig.show()
    """
    import plotly.graph_objects as go

    if label_names is None:
        label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}

    if not contributions:
        return go.Figure()

    tokens = [c[0] for c in contributions]
    values = [c[1] for c in contributions]

    # Colour: green for positive SHAP (pushes toward class), red for negative
    colors = ["#00CC96" if v > 0 else "#EF553B" for v in values]

    # Display in ascending order so most-impactful bar is at the top
    tokens_rev = tokens[::-1]
    values_rev = values[::-1]
    colors_rev = colors[::-1]

    class_name = label_names.get(predicted_class, str(predicted_class))
    fig = go.Figure(go.Bar(
        x=values_rev,
        y=tokens_rev,
        orientation="h",
        marker_color=colors_rev,
        text=[f"{v:+.3f}" for v in values_rev],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Token Contributions → Predicted: {class_name}",
        title_x=0.5,
        xaxis_title="SHAP Value",
        yaxis_title="Token",
        height=max(320, len(contributions) * 28),
        margin=dict(l=120, r=60, t=60, b=40),
        xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="black"),
    )
    return fig
