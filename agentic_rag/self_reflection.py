"""
agentic_rag/self_reflection.py — Self-reflection scoring loop for Agentic RAG.

Asks the LLM to score how well a set of retrieved documents answers a query,
then decides whether to accept the results or request another rewrite-retrieve
cycle.  The scorer returns a numeric confidence in [0, 1] and a brief rationale.

Key exports:
    ReflectionResult     — dataclass holding score, rationale, and accept flag
    SelfReflector        — class that wraps the LLM scoring call
    reflect_on_results   — module-level convenience function
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import get_logger  # noqa: E402

logger = get_logger(__name__)

_SCORE_THRESHOLD = 0.7  # accept results when score >= this value

_SYSTEM_PROMPT = (
    "You are a retrieval quality evaluator for a sentiment-analysis RAG system.  "
    "Given a user query and a list of retrieved documents, assess how well the "
    "documents answer the query.\n\n"
    "Return a JSON object with exactly two keys:\n"
    '  "score": a float in [0.0, 1.0] — 0 means completely irrelevant, '
    "1 means perfectly relevant\n"
    '  "rationale": one sentence explaining the score\n\n'
    "Return ONLY the JSON object — no markdown fences, no extra text."
)


@dataclass(frozen=True)
class ReflectionResult:
    """Outcome of a single self-reflection evaluation.

    Attributes:
        score: Relevance score in ``[0.0, 1.0]``.
        rationale: One-sentence explanation from the LLM.
        accepted: ``True`` when ``score >= _SCORE_THRESHOLD`` (0.7).
    """

    score: float
    rationale: str
    accepted: bool


class SelfReflector:
    """Score retrieved documents against a query and decide whether to accept.

    Uses the OpenAI-compatible endpoint configured in ``agentic_rag.config``.
    Parses a structured JSON response from the LLM; falls back to a default
    low score if the response cannot be parsed.

    Args:
        threshold: Minimum score to accept results.  Defaults to 0.7.
        temperature: Sampling temperature.  Defaults to
                     ``RAG_CONFIG.reflect_temperature``.

    Example:
        >>> reflector = SelfReflector()
        >>> docs = [{"clean_text": "I hate the new UI", "label": 0}]
        >>> result = reflector.reflect("negative product feedback", docs)
        >>> isinstance(result.score, float)
        True
    """

    def __init__(
        self,
        threshold: float = _SCORE_THRESHOLD,
        temperature: float | None = None,
    ) -> None:
        from openai import OpenAI
        from agentic_rag.config import RAG_CONFIG

        self._cfg = RAG_CONFIG
        self._threshold = threshold
        self._temperature = (
            temperature if temperature is not None
            else RAG_CONFIG.reflect_temperature
        )
        self._client = OpenAI(
            api_key=RAG_CONFIG.openai_api_key,
            base_url=RAG_CONFIG.base_url,
        )
        self._model = RAG_CONFIG.model
        logger.info(
            "SelfReflector ready — model=%s  threshold=%.2f  temperature=%.2f",
            self._model, self._threshold, self._temperature,
        )

    def reflect(
        self,
        query: str,
        documents: List[Dict],
        text_key: str = "clean_text",
    ) -> ReflectionResult:
        """Score how well ``documents`` answer ``query``.

        Args:
            query: The (possibly rewritten) retrieval query.
            documents: List of document dicts from FAISS search results.
                       Each dict must contain the ``text_key`` field.
            text_key: Key in each document dict containing the text to evaluate.
                      Defaults to ``'clean_text'``.

        Returns:
            :class:`ReflectionResult` with ``score``, ``rationale``, and
            ``accepted`` fields.

        Raises:
            ValueError: When ``query`` is empty or ``documents`` is empty.

        Example:
            >>> reflector = SelfReflector()
            >>> result = reflector.reflect("users love product X",
            ...     [{"clean_text": "Product X is amazing", "label": 1}])
            >>> 0.0 <= result.score <= 1.0
            True
        """
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string.")
        if not documents:
            raise ValueError("documents must be a non-empty list.")

        snippets = "\n".join(
            f"- {doc.get(text_key, '')}" for doc in documents[:5]
        )
        user_content = (
            f"Query: {query}\n\n"
            f"Retrieved documents:\n{snippets}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content.strip()
        score, rationale = self._parse_response(raw)
        accepted = score >= self._threshold

        logger.info(
            "Reflection score=%.3f  accepted=%s  rationale=%r",
            score, accepted, rationale,
        )
        return ReflectionResult(score=score, rationale=rationale, accepted=accepted)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> tuple[float, str]:
        """Parse the LLM JSON response into (score, rationale).

        Falls back to ``(0.0, 'parse error')`` rather than raising so the
        pipeline can decide whether to retry.

        Args:
            raw: Raw string returned by the LLM.

        Returns:
            Tuple ``(score, rationale)`` where score is clamped to [0, 1].
        """
        try:
            # Strip optional markdown code fences the model might add
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(cleaned)
            score = float(data["score"])
            score = max(0.0, min(1.0, score))  # clamp to [0, 1]
            rationale = str(data.get("rationale", ""))
            return score, rationale
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse reflection response: %s — raw=%r", exc, raw)
            return 0.0, "parse error"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def reflect_on_results(
    query: str,
    documents: List[Dict],
    threshold: float = _SCORE_THRESHOLD,
    temperature: float | None = None,
) -> ReflectionResult:
    """Score retrieved documents against a query using a fresh SelfReflector.

    Convenience wrapper — creates a SelfReflector and immediately calls
    ``reflect()``.  Prefer constructing a ``SelfReflector`` instance directly
    when making multiple calls to amortise client initialisation cost.

    Args:
        query: The retrieval query.
        documents: Retrieved document dicts.
        threshold: Acceptance threshold.  Defaults to 0.7.
        temperature: Sampling temperature override.

    Returns:
        :class:`ReflectionResult` with score, rationale, and accepted flag.

    Example:
        >>> result = reflect_on_results("positive reviews",
        ...     [{"clean_text": "Great product!", "label": 1}])
        >>> isinstance(result, ReflectionResult)
        True
    """
    return SelfReflector(
        threshold=threshold, temperature=temperature
    ).reflect(query, documents)
