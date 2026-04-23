"""
agentic_rag/tests/test_embedding_utils.py — Unit tests for BertEmbedder.

Tests cover:
  - Output shape for single and batch inputs
  - Output dtype (float32)
  - Determinism (same text → same vector)
  - Empty-input guard
  - Cross-batch consistency (encode_batch with batch_size=1 vs batch_size=32)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on the path before importing agentic_rag modules
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping BERT tests.")


# ---------------------------------------------------------------------------
# Fixture — load BertEmbedder once per session to save GPU memory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedder():
    """Return a single BertEmbedder instance shared across all tests."""
    from agentic_rag.embedding_utils import BertEmbedder
    return BertEmbedder()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_embedding_shape(embedder):
    """get_embedding() returns a 1-D array of length 768."""
    vec = embedder.get_embedding("This is a test sentence.")
    assert vec.shape == (768,), f"Expected (768,), got {vec.shape}"


def test_get_embedding_dtype(embedder):
    """get_embedding() returns float32."""
    vec = embedder.get_embedding("Check dtype.")
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


def test_encode_batch_shape(embedder):
    """encode_batch() returns shape (n, 768) for n inputs."""
    texts = ["Hello world", "Sentiment analysis rocks", "Third sentence here"]
    vecs = embedder.encode_batch(texts)
    assert vecs.shape == (3, 768), f"Expected (3, 768), got {vecs.shape}"


def test_encode_batch_dtype(embedder):
    """encode_batch() returns float32 matrix."""
    vecs = embedder.encode_batch(["dtype check"])
    assert vecs.dtype == np.float32


def test_determinism(embedder):
    """Same input text produces identical vectors across two calls."""
    text = "Reproducibility check sentence"
    v1 = embedder.get_embedding(text)
    v2 = embedder.get_embedding(text)
    np.testing.assert_array_equal(v1, v2)


def test_encode_batch_single_matches_get_embedding(embedder):
    """encode_batch([text])[0] is identical to get_embedding(text)."""
    text = "Consistency between single and batch"
    single = embedder.get_embedding(text)
    batch = embedder.encode_batch([text])
    np.testing.assert_array_equal(single, batch[0])


def test_encode_batch_batchsize_consistency(embedder):
    """encode_batch produces the same result regardless of batch_size."""
    texts = [
        "First sentence for batch consistency.",
        "Second sentence here.",
        "Third one to fill the batch.",
    ]
    vecs_bs1 = embedder.encode_batch(texts, batch_size=1)
    vecs_bs32 = embedder.encode_batch(texts, batch_size=32)
    np.testing.assert_array_almost_equal(vecs_bs1, vecs_bs32, decimal=5)


def test_empty_input_raises(embedder):
    """encode_batch([]) raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        embedder.encode_batch([])


def test_embedding_dim_constant():
    """BertEmbedder.EMBEDDING_DIM equals 768."""
    from agentic_rag.embedding_utils import BertEmbedder
    assert BertEmbedder.EMBEDDING_DIM == 768
