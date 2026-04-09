"""
src/preprocess.py — Advanced text preprocessing utilities.

Provides tokenisation, stopword removal, lemmatisation, and feature
engineering helpers used by both baseline and deep-learning pipelines.
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# NLTK resource download (safe / idempotent)
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present.

    Handles both legacy ``punkt`` (NLTK < 3.9) and the renamed
    ``punkt_tab`` tokeniser models (NLTK >= 3.9).
    """
    try:
        import nltk

        corpora = ("stopwords", "wordnet", "omw-1.4")
        for resource in corpora:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource, quiet=True)

        # punkt_tab (NLTK >= 3.9) supersedes punkt — try both
        for tok_resource in ("punkt_tab", "punkt"):
            try:
                nltk.data.find(f"tokenizers/{tok_resource}")
                break  # found one, no need to check the other
            except LookupError:
                try:
                    nltk.download(tok_resource, quiet=True)
                    break
                except Exception:
                    continue  # try next

    except ImportError:
        logger.warning("NLTK not installed — skipping resource download.")


_ensure_nltk_data()


# ---------------------------------------------------------------------------
# Tokenise & normalise
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenise a string into a list of word tokens.

    Uses NLTK ``word_tokenize`` when available; falls back to a simple
    whitespace split otherwise.

    Args:
        text: Pre-cleaned input string.

    Returns:
        List of token strings.

    Example:
        >>> tokenize("machine learning is great")
        ['machine', 'learning', 'is', 'great']
    """
    if not isinstance(text, str):
        return []
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        return text.split()


def remove_stopwords(tokens: List[str], language: str = "english") -> List[str]:
    """Remove stopwords from a token list.

    Args:
        tokens: List of string tokens.
        language: NLTK stopword language (default ``'english'``).

    Returns:
        Filtered list with stopwords removed.

    Example:
        >>> remove_stopwords(['this', 'is', 'great'])
        ['great']
    """
    try:
        from nltk.corpus import stopwords
        stop_set = set(stopwords.words(language))
        return [t for t in tokens if t.lower() not in stop_set]
    except Exception:
        return tokens


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatise a list of tokens using NLTK ``WordNetLemmatizer``.

    Args:
        tokens: List of string tokens.

    Returns:
        List of lemmatised tokens.

    Example:
        >>> lemmatize(['running', 'better', 'dogs'])
        ['running', 'better', 'dog']
    """
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]
    except Exception:
        return tokens


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def add_text_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Add derived text-length and lexical diversity features to a DataFrame.

    Adds columns:
    * ``word_count`` — number of whitespace-separated tokens
    * ``char_count`` — number of characters
    * ``avg_word_len`` — average word length
    * ``unique_word_ratio`` — unique tokens / total tokens

    Args:
        df: Input DataFrame with a *text_col* column.
        text_col: Name of the text column (default ``'clean_text'``).

    Returns:
        DataFrame with four new feature columns appended.

    Example:
        >>> df = add_text_features(df)
        >>> df[['word_count', 'char_count']].describe()
    """
    df = df.copy()
    tokens_series = df[text_col].str.split()

    df["word_count"] = tokens_series.str.len().fillna(0).astype(int)
    df["char_count"] = df[text_col].str.len().fillna(0).astype(int)
    df["avg_word_len"] = df[text_col].apply(
        lambda t: (sum(len(w) for w in t.split()) / max(len(t.split()), 1))
        if isinstance(t, str) else 0.0
    )
    df["unique_word_ratio"] = tokens_series.apply(
        lambda toks: len(set(toks)) / max(len(toks), 1) if isinstance(toks, list) else 0.0
    )
    logger.info("Text features added to DataFrame.")
    return df
