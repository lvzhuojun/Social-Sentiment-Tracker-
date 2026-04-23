"""
agentic_rag/build_index.py — Build and persist a FAISS index from sentiment data.

Reads the processed training CSV, encodes each row's ``clean_text`` column
with BertEmbedder, and writes a FAISS IndexFlatIP (inner-product / cosine after
L2-normalisation) to disk alongside a pickle mapping integer IDs → row dicts.

The index is stored in ``agentic_rag/vector_store/`` which is git-ignored.

Usage (run from repo root):
    python -m agentic_rag.build_index
    python -m agentic_rag.build_index --csv data/processed/train.csv --batch 64

Key exports:
    build_index()   — build and save index from a DataFrame
    load_index()    — load a previously built index from disk
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import PROCESSED_DATA_DIR, get_logger  # noqa: E402

logger = get_logger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss not installed — build_index disabled.")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(
    csv_path: Path | None = None,
    text_col: str = "clean_text",
    batch_size: int = 32,
) -> Tuple["faiss.Index", List[Dict]]:
    """Encode all rows in a CSV and build a FAISS IndexFlatIP.

    Embeddings are L2-normalised before insertion so that inner-product
    search is equivalent to cosine similarity.

    Args:
        csv_path: Path to the processed CSV file.  Defaults to
                  ``data/processed/train.csv``.
        text_col: Name of the column containing pre-cleaned text.
                  Defaults to ``'clean_text'``.
        batch_size: Embedding batch size passed to BertEmbedder.
                    Defaults to 32.

    Returns:
        Tuple ``(index, id_map)`` where:
        * ``index`` — trained ``faiss.IndexFlatIP`` containing all vectors.
        * ``id_map`` — list of row dicts (``id``, ``text``, ``label``,
          ``clean_text``) indexed by FAISS integer ID.

    Raises:
        ImportError: When ``faiss`` is not installed.
        FileNotFoundError: When ``csv_path`` does not exist.

    Example:
        >>> index, id_map = build_index()
        >>> index.ntotal > 0
        True
    """
    if not _FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required.  pip install faiss-cpu")
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas is required.  pip install pandas")

    csv_path = Path(csv_path) if csv_path else PROCESSED_DATA_DIR / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}.  "
            "Run scripts/train_full.py first to generate processed splits."
        )

    logger.info("Reading %s …", csv_path)
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {csv_path}.  "
                         f"Available columns: {list(df.columns)}")

    texts = df[text_col].fillna("").tolist()
    logger.info("Encoding %d texts with BertEmbedder …", len(texts))

    from agentic_rag.embedding_utils import BertEmbedder
    embedder = BertEmbedder(batch_size=batch_size)
    embeddings = embedder.encode_batch(texts)  # (n, 768)

    # L2-normalise so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = (embeddings / norms).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built — %d vectors, dim=%d", index.ntotal, dim)

    # Build id_map: FAISS int ID (row position) → original row dict
    keep_cols = [c for c in ("id", "text", "clean_text", "label") if c in df.columns]
    id_map: List[Dict] = df[keep_cols].to_dict(orient="records")

    # Persist to disk
    from agentic_rag.config import RAG_CONFIG
    RAG_CONFIG.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(RAG_CONFIG.index_path))
    with open(RAG_CONFIG.id_map_path, "wb") as fh:
        pickle.dump(id_map, fh)

    logger.info("Index saved → %s", RAG_CONFIG.index_path)
    logger.info("ID map saved → %s", RAG_CONFIG.id_map_path)
    return index, id_map


def load_index() -> Tuple["faiss.Index", List[Dict]]:
    """Load a previously built FAISS index and its ID map from disk.

    Returns:
        Tuple ``(index, id_map)`` ready for similarity search.

    Raises:
        FileNotFoundError: When the index or ID map file does not exist.
                           Call ``build_index()`` first.
        ImportError: When ``faiss`` is not installed.

    Example:
        >>> index, id_map = load_index()
        >>> index.ntotal > 0
        True
    """
    if not _FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is required.  pip install faiss-cpu")

    from agentic_rag.config import RAG_CONFIG

    if not RAG_CONFIG.index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {RAG_CONFIG.index_path}.  "
            "Run: python -m agentic_rag.build_index"
        )
    if not RAG_CONFIG.id_map_path.exists():
        raise FileNotFoundError(
            f"ID map not found at {RAG_CONFIG.id_map_path}.  "
            "Run: python -m agentic_rag.build_index"
        )

    index = faiss.read_index(str(RAG_CONFIG.index_path))
    with open(RAG_CONFIG.id_map_path, "rb") as fh:
        id_map: List[Dict] = pickle.load(fh)

    logger.info("Index loaded — %d vectors", index.ntotal)
    return index, id_map


def search(
    query_vec: np.ndarray,
    index: "faiss.Index",
    id_map: List[Dict],
    top_k: int | None = None,
) -> List[Dict]:
    """Return the top-k most similar documents for a query embedding.

    Args:
        query_vec: Float32 array of shape ``(768,)`` — must already be
                   L2-normalised (same as embeddings in the index).
        index: Loaded FAISS index.
        id_map: List mapping FAISS integer IDs to document dicts.
        top_k: Number of results to return.  Defaults to
               ``RAG_CONFIG.top_k``.

    Returns:
        List of document dicts, each augmented with a ``score`` key
        (cosine similarity, range ``[-1, 1]``).  Ordered by descending score.

    Example:
        >>> results = search(query_vec, index, id_map)
        >>> results[0]["score"] >= results[-1]["score"]
        True
    """
    from agentic_rag.config import RAG_CONFIG
    k = top_k if top_k is not None else RAG_CONFIG.top_k

    vec = query_vec.astype(np.float32).reshape(1, -1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    scores, indices = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(id_map):
            continue
        doc = dict(id_map[idx])
        doc["score"] = float(score)
        results.append(doc)
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from sentiment CSV.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Path to processed CSV (default: data/processed/train.csv)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Embedding batch size (default: 32)")
    args = parser.parse_args()
    build_index(csv_path=args.csv, batch_size=args.batch)
