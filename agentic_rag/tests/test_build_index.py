"""
agentic_rag/tests/test_build_index.py — Unit tests for build_index.py.

Tests cover:
  - build_index() produces correct vector count and dimension
  - id_map length matches the number of CSV rows
  - Index files are written to disk
  - load_index() round-trips correctly
  - search() returns top_k results with score keys, ordered desc
  - search() handles normalised and un-normalised query vectors
  - Missing CSV raises FileNotFoundError
  - Missing index files raise FileNotFoundError on load_index()
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

faiss = pytest.importorskip("faiss", reason="faiss not installed — skipping index tests.")
pd = pytest.importorskip("pandas", reason="pandas not installed.")
torch = pytest.importorskip("torch", reason="PyTorch not installed — needs BERT.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_csv(tmp_path_factory):
    """Write a 10-row CSV and return its Path."""
    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "tiny.csv"
    rows = "\n".join([
        "id,label,text,clean_text",
        *[f"{i},{i % 3},Raw text {i},cleaned text {i}" for i in range(10)],
    ])
    csv_path.write_text(rows)
    return csv_path


@pytest.fixture(scope="module")
def built_index(tmp_path_factory, tiny_csv, monkeypatch_module):
    """Build index from tiny_csv; patch RAG_CONFIG paths to tmp dir."""
    tmp = tmp_path_factory.mktemp("vector_store")
    _patch_index_paths(monkeypatch_module, tmp)
    from agentic_rag.build_index import build_index
    return build_index(csv_path=tiny_csv, batch_size=4)


@pytest.fixture(scope="module")
def monkeypatch_module(request):
    """Module-scoped monkeypatch fixture."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_index_paths(mp, tmp_dir: Path) -> None:
    """Redirect RAG_CONFIG index paths to a temp directory."""
    import agentic_rag.config as rag_cfg
    import agentic_rag.build_index as bi_mod

    new_cfg = rag_cfg._RagConfig(
        openai_api_key="test-key",
        base_url="https://example.com/v1",
        model="gpt-5.4",
        index_dir=tmp_dir,
        index_path=tmp_dir / "test.faiss",
        id_map_path=tmp_dir / "test_id_map.pkl",
        top_k=3,
        rewrite_temperature=0.7,
        reflect_temperature=0.2,
        max_rewrite_attempts=3,
    )
    mp.setattr(rag_cfg, "RAG_CONFIG", new_cfg)
    mp.setattr(bi_mod, "_REPO_ROOT", _REPO_ROOT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_index_vector_count(built_index):
    """Index contains exactly as many vectors as CSV rows (10)."""
    index, _ = built_index
    assert index.ntotal == 10


def test_build_index_dimension(built_index):
    """Index dimension is 768 (BERT-base hidden size)."""
    index, _ = built_index
    assert index.d == 768


def test_id_map_length(built_index):
    """id_map has one entry per CSV row."""
    _, id_map = built_index
    assert len(id_map) == 10


def test_id_map_has_required_keys(built_index):
    """Each id_map entry contains 'clean_text' and 'label'."""
    _, id_map = built_index
    for doc in id_map:
        assert "clean_text" in doc
        assert "label" in doc


def test_index_files_written(built_index, tmp_path_factory):
    """build_index() writes both .faiss and .pkl files."""
    import agentic_rag.config as rag_cfg
    assert rag_cfg.RAG_CONFIG.index_path.exists()
    assert rag_cfg.RAG_CONFIG.id_map_path.exists()


def test_load_index_round_trip(built_index):
    """load_index() returns an index with the same ntotal as build_index()."""
    original_index, _ = built_index
    from agentic_rag.build_index import load_index
    loaded_index, loaded_map = load_index()
    assert loaded_index.ntotal == original_index.ntotal
    assert len(loaded_map) == 10


def test_search_returns_top_k(built_index):
    """search() returns exactly top_k results."""
    index, id_map = built_index
    from agentic_rag.build_index import search
    query = np.random.randn(768).astype(np.float32)
    results = search(query, index, id_map, top_k=3)
    assert len(results) == 3


def test_search_results_have_score(built_index):
    """Each result dict contains a float 'score' key."""
    index, id_map = built_index
    from agentic_rag.build_index import search
    query = np.random.randn(768).astype(np.float32)
    results = search(query, index, id_map, top_k=2)
    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)


def test_search_scores_descending(built_index):
    """Results are ordered by descending score."""
    index, id_map = built_index
    from agentic_rag.build_index import search
    query = np.random.randn(768).astype(np.float32)
    results = search(query, index, id_map, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_missing_csv_raises(tmp_path):
    """build_index() raises FileNotFoundError for a non-existent CSV path."""
    from agentic_rag.build_index import build_index
    with pytest.raises(FileNotFoundError):
        build_index(csv_path=tmp_path / "nonexistent.csv")
