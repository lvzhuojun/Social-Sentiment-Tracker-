"""
agentic_rag/tests/test_self_reflection.py — Unit tests for self_reflection.py.

All LLM calls are mocked.  Tests cover:
  - Correct JSON parsing into ReflectionResult fields
  - Score clamping to [0, 1]
  - accepted=True when score >= threshold
  - accepted=False when score < threshold
  - Graceful fallback on malformed JSON (score=0, rationale='parse error')
  - ValueError on empty query or empty documents list
  - reflect_on_results() convenience wrapper delegates correctly
  - Markdown code fences are stripped before JSON parsing
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DOCS = [{"clean_text": "I love this product", "label": 1}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_reflector(mock_create, threshold: float = 0.7):
    from agentic_rag.self_reflection import SelfReflector
    reflector = SelfReflector.__new__(SelfReflector)
    reflector._threshold = threshold
    reflector._temperature = 0.2
    reflector._model = "gpt-5.4"
    reflector._client = MagicMock()
    reflector._client.chat.completions.create = mock_create
    return reflector


def _json_resp(score: float, rationale: str = "ok") -> str:
    return json.dumps({"score": score, "rationale": rationale})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reflect_returns_reflection_result():
    """reflect() returns a ReflectionResult instance."""
    from agentic_rag.self_reflection import ReflectionResult
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(0.8)))
    reflector = _make_reflector(mock_create)
    result = reflector.reflect("positive sentiment", DOCS)
    assert isinstance(result, ReflectionResult)


def test_reflect_score_parsed_correctly():
    """score field matches the value in the JSON response."""
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(0.85)))
    reflector = _make_reflector(mock_create)
    result = reflector.reflect("query", DOCS)
    assert abs(result.score - 0.85) < 1e-6


def test_reflect_accepted_true_above_threshold():
    """accepted is True when score >= threshold (0.7)."""
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(0.9)))
    reflector = _make_reflector(mock_create, threshold=0.7)
    result = reflector.reflect("query", DOCS)
    assert result.accepted is True


def test_reflect_accepted_false_below_threshold():
    """accepted is False when score < threshold."""
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(0.5)))
    reflector = _make_reflector(mock_create, threshold=0.7)
    result = reflector.reflect("query", DOCS)
    assert result.accepted is False


def test_reflect_score_clamped_above_one():
    """Scores > 1.0 in the response are clamped to 1.0."""
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(1.5)))
    reflector = _make_reflector(mock_create)
    result = reflector.reflect("query", DOCS)
    assert result.score == 1.0


def test_reflect_score_clamped_below_zero():
    """Scores < 0.0 in the response are clamped to 0.0."""
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(-0.3)))
    reflector = _make_reflector(mock_create)
    result = reflector.reflect("query", DOCS)
    assert result.score == 0.0


def test_reflect_malformed_json_fallback():
    """Malformed JSON produces score=0.0 and rationale='parse error'."""
    mock_create = MagicMock(return_value=_make_mock_response("not valid json {{"))
    reflector = _make_reflector(mock_create)
    result = reflector.reflect("query", DOCS)
    assert result.score == 0.0
    assert "parse error" in result.rationale


def test_reflect_markdown_fence_stripped():
    """JSON wrapped in ```json ... ``` fences is parsed correctly."""
    fenced = "```json\n" + _json_resp(0.75) + "\n```"
    mock_create = MagicMock(return_value=_make_mock_response(fenced))
    reflector = _make_reflector(mock_create)
    result = reflector.reflect("query", DOCS)
    assert abs(result.score - 0.75) < 1e-6


def test_reflect_empty_query_raises():
    """reflect() raises ValueError for empty query."""
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(0.9)))
    reflector = _make_reflector(mock_create)
    with pytest.raises(ValueError, match="non-empty"):
        reflector.reflect("", DOCS)


def test_reflect_empty_documents_raises():
    """reflect() raises ValueError for empty documents list."""
    mock_create = MagicMock(return_value=_make_mock_response(_json_resp(0.9)))
    reflector = _make_reflector(mock_create)
    with pytest.raises(ValueError, match="non-empty"):
        reflector.reflect("query", [])


def test_reflect_on_results_convenience_function():
    """reflect_on_results() delegates to SelfReflector.reflect()."""
    from agentic_rag.self_reflection import ReflectionResult

    dummy_result = ReflectionResult(score=0.8, rationale="good", accepted=True)

    with patch("agentic_rag.self_reflection.SelfReflector") as mock_cls:
        instance = MagicMock()
        instance.reflect.return_value = dummy_result
        mock_cls.return_value = instance

        from agentic_rag.self_reflection import reflect_on_results
        result = reflect_on_results("query", DOCS, threshold=0.6, temperature=0.1)

        mock_cls.assert_called_once_with(threshold=0.6, temperature=0.1)
        instance.reflect.assert_called_once_with("query", DOCS)
        assert result is dummy_result
