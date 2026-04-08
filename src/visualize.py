"""
src/visualize.py — Interactive visualisation functions using Plotly.

All chart functions return a Plotly Figure so they can be rendered in
notebooks, Streamlit, or saved to disk. Word-cloud images are saved as PNGs.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FIGURES_DIR, get_logger

logger = get_logger(__name__)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette — consistent across charts
_PALETTE = {0: "#EF553B", 1: "#00CC96", 2: "#636EFA"}   # neg, pos, neutral
_LABEL_NAMES = {0: "Negative", 1: "Positive", 2: "Neutral"}


# ---------------------------------------------------------------------------
# Sentiment distribution — pie chart
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(df: pd.DataFrame, label_col: str = "label") -> go.Figure:
    """Interactive pie chart of sentiment class distribution.

    Args:
        df: DataFrame containing a ``label`` column.
        label_col: Name of the label column (default ``'label'``).

    Returns:
        :class:`plotly.graph_objects.Figure`.

    Example:
        >>> fig = plot_sentiment_distribution(df)
        >>> fig.show()
    """
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["sentiment"] = counts["label"].map(_LABEL_NAMES).fillna(counts["label"].astype(str))
    colors = [_PALETTE.get(int(lbl), "#AB63FA") for lbl in counts["label"]]

    fig = px.pie(
        counts,
        values="count",
        names="sentiment",
        color_discrete_sequence=colors,
        title="Sentiment Distribution",
        hole=0.35,
    )
    fig.update_traces(textinfo="percent+label", pull=[0.03] * len(counts))
    fig.update_layout(legend_title="Sentiment", title_x=0.5)
    logger.info("Sentiment distribution pie chart created.")
    return fig


# ---------------------------------------------------------------------------
# Text length distribution — histogram by sentiment
# ---------------------------------------------------------------------------

def plot_text_length_distribution(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
) -> go.Figure:
    """Overlapping histogram of text lengths grouped by sentiment class.

    Args:
        df: DataFrame with *text_col* and *label_col* columns.
        text_col: Column containing the text strings.
        label_col: Column containing integer labels.

    Returns:
        :class:`plotly.graph_objects.Figure`.

    Example:
        >>> fig = plot_text_length_distribution(df)
        >>> fig.show()
    """
    df = df.copy()
    df["text_length"] = df[text_col].str.split().str.len()
    df["sentiment"] = df[label_col].map(_LABEL_NAMES).fillna(df[label_col].astype(str))

    fig = px.histogram(
        df,
        x="text_length",
        color="sentiment",
        barmode="overlay",
        nbins=50,
        opacity=0.7,
        color_discrete_map={v: _PALETTE.get(k, "#AB63FA") for k, v in _LABEL_NAMES.items()},
        title="Text Length Distribution by Sentiment",
        labels={"text_length": "Word Count"},
    )
    fig.update_layout(bargap=0.05, title_x=0.5)
    logger.info("Text length distribution chart created.")
    return fig


# ---------------------------------------------------------------------------
# Word cloud — saved as PNG
# ---------------------------------------------------------------------------

def plot_wordcloud(
    df: pd.DataFrame,
    sentiment: int = 1,
    text_col: str = "clean_text",
    label_col: str = "label",
    max_words: int = 200,
) -> Path:
    """Generate and save a word cloud for a given sentiment class.

    Args:
        df: DataFrame with *text_col* and *label_col* columns.
        sentiment: Label value to filter on (default ``1`` = positive).
        text_col: Column with text data.
        label_col: Column with labels.
        max_words: Maximum number of words in the cloud (default 200).

    Returns:
        Path to the saved PNG file.

    Raises:
        ImportError: If ``wordcloud`` or ``matplotlib`` are not installed.

    Example:
        >>> path = plot_wordcloud(df, sentiment=0)
    """
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install wordcloud: pip install wordcloud") from exc

    subset = df[df[label_col] == sentiment][text_col]
    if subset.empty:
        logger.warning("No data for sentiment label %d — wordcloud skipped.", sentiment)
        return FIGURES_DIR / "wordcloud_empty.png"

    text_corpus = " ".join(subset.dropna())
    colormap = "YlOrRd" if sentiment == 0 else "YlGn" if sentiment == 1 else "Blues"

    wc = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color="white",
        colormap=colormap,
        collocations=False,
    ).generate(text_corpus)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    label_name = _LABEL_NAMES.get(sentiment, str(sentiment))
    ax.set_title(f"Word Cloud — {label_name} Sentiment", fontsize=14)
    plt.tight_layout()

    save_path = FIGURES_DIR / f"wordcloud_{label_name.lower()}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Word cloud saved to %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Sentiment over time — line chart
# ---------------------------------------------------------------------------

def plot_sentiment_over_time(
    df: pd.DataFrame,
    date_col: str = "date",
    label_col: str = "label",
    freq: str = "D",
) -> go.Figure:
    """Interactive time-series line chart of daily sentiment counts.

    Args:
        df: DataFrame with *date_col* and *label_col* columns.
        date_col: Column containing date strings or :class:`datetime` objects.
        label_col: Column containing integer labels.
        freq: Resampling frequency (``'D'`` = daily, ``'W'`` = weekly).

    Returns:
        :class:`plotly.graph_objects.Figure`.

    Example:
        >>> fig = plot_sentiment_over_time(df)
        >>> fig.show()
    """
    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        logger.warning("Could not parse date column '%s'.", date_col)
        return go.Figure()

    df = df.dropna(subset=[date_col])
    df["sentiment"] = df[label_col].map(_LABEL_NAMES).fillna(df[label_col].astype(str))

    time_df = (
        df.groupby([pd.Grouper(key=date_col, freq=freq), "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        time_df,
        x=date_col,
        y="count",
        color="sentiment",
        color_discrete_map={v: _PALETTE.get(k, "#AB63FA") for k, v in _LABEL_NAMES.items()},
        title="Sentiment Trend Over Time",
        labels={date_col: "Date", "count": "Number of Posts"},
        markers=True,
    )
    fig.update_layout(title_x=0.5, legend_title="Sentiment")
    logger.info("Sentiment over time chart created.")
    return fig


# ---------------------------------------------------------------------------
# Top keywords — horizontal bar chart (TF-IDF weighted)
# ---------------------------------------------------------------------------

def plot_top_keywords(
    df: pd.DataFrame,
    n: int = 20,
    text_col: str = "clean_text",
    label_col: str = "label",
    sentiment: int | None = None,
) -> go.Figure:
    """Horizontal bar chart of the top-N TF-IDF keywords.

    Args:
        df: DataFrame with *text_col* column.
        n: Number of top keywords to display (default 20).
        text_col: Column with pre-cleaned text.
        label_col: Column with labels (used when *sentiment* is set).
        sentiment: If provided, filter to this label before computing keywords.

    Returns:
        :class:`plotly.graph_objects.Figure`.

    Example:
        >>> fig = plot_top_keywords(df, n=15, sentiment=0)
        >>> fig.show()
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    subset = df.copy()
    if sentiment is not None:
        subset = subset[subset[label_col] == sentiment]
        title = f"Top {n} Keywords — {_LABEL_NAMES.get(sentiment, str(sentiment))} Sentiment"
    else:
        title = f"Top {n} Keywords (All Sentiments)"

    texts = subset[text_col].dropna().tolist()
    if not texts:
        logger.warning("No texts available for keyword extraction.")
        return go.Figure()

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    vocab = vectorizer.get_feature_names_out()

    top_idx = np.argsort(mean_scores)[-n:][::-1]
    keywords = vocab[top_idx]
    scores = mean_scores[top_idx]

    # Sort ascending so the longest bar is at the top
    sorted_order = np.argsort(scores)
    keywords = keywords[sorted_order]
    scores = scores[sorted_order]

    color = _PALETTE.get(sentiment, "#636EFA") if sentiment is not None else "#636EFA"

    fig = go.Figure(go.Bar(
        x=scores,
        y=keywords,
        orientation="h",
        marker_color=color,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Mean TF-IDF Score",
        yaxis_title="Keyword",
        height=max(400, n * 25),
        margin=dict(l=160, r=40, t=60, b=40),
    )
    logger.info("Top keywords chart created (%d keywords).", n)
    return fig


# ---------------------------------------------------------------------------
# Gauge chart helper (used by Streamlit demo)
# ---------------------------------------------------------------------------

def plot_confidence_gauge(confidence: float, sentiment_label: str) -> go.Figure:
    """Circular gauge showing prediction confidence.

    Args:
        confidence: Float in [0, 1].
        sentiment_label: Display string (e.g. ``'Positive'``).

    Returns:
        :class:`plotly.graph_objects.Figure`.

    Example:
        >>> fig = plot_confidence_gauge(0.87, "Positive")
    """
    color = "#00CC96" if "Positive" in sentiment_label else "#EF553B"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%"},
        title={"text": f"Confidence — {sentiment_label}", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 50], "color": "#FEE0D2"},
                {"range": [50, 75], "color": "#FCBBA1"},
                {"range": [75, 100], "color": "#FC9272"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": confidence * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig
