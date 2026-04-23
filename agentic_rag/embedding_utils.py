"""
agentic_rag/embedding_utils.py — BERT [CLS] embedding extractor for Agentic RAG.

Wraps the existing fine-tuned SentimentClassifier (src/bert_model.py) to expose
768-dimensional sentence embeddings from the [CLS] token of the final hidden layer.
The underlying model weights and tokeniser are reused without modification.

Key exports:
    BertEmbedder  — class providing get_embedding() and encode_batch()
"""

import sys
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — makes social-sentiment-tracker root importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import MAX_LENGTH, get_logger  # noqa: E402

logger = get_logger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available — BertEmbedder disabled.")


class BertEmbedder:
    """Extract 768-dim [CLS] embeddings from the fine-tuned BERT model.

    Reuses the SentimentClassifier loaded via src/bert_model.load_bert_model()
    without modifying the original file.  The [CLS] token from the final
    hidden layer of bert-base-uncased is used as the sentence representation.

    Args:
        model_path: Path to the ``.pt`` checkpoint.  Defaults to
                    ``config.BERT_MODEL_PATH``.
        batch_size: Texts processed per forward pass.  Defaults to 32.
        max_length: Tokeniser maximum sequence length.  Defaults to
                    ``config.MAX_LENGTH``.

    Example:
        >>> embedder = BertEmbedder()
        >>> vec = embedder.get_embedding("Great product!")
        >>> vec.shape
        (768,)
    """

    EMBEDDING_DIM: int = 768

    def __init__(
        self,
        model_path: Path | None = None,
        batch_size: int = 32,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for BertEmbedder. "
                "Install it with: pip install torch"
            )

        from src.bert_model import load_bert_model

        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model, self._tokenizer = load_bert_model(path=model_path)
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            "BertEmbedder ready — device=%s  batch_size=%d  max_length=%d",
            self._device, self._batch_size, self._max_length,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding(self, text: str) -> np.ndarray:
        """Encode a single string to a 768-dimensional float32 vector.

        Args:
            text: Input string to embed.

        Returns:
            Float32 numpy array of shape ``(768,)``.

        Example:
            >>> embedder = BertEmbedder()
            >>> vec = embedder.get_embedding("I love this!")
            >>> vec.shape
            (768,)
        """
        return self.encode_batch([text])[0]

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a list of strings to a 2-D float32 embedding matrix.

        Args:
            texts: List of input strings.  Must be non-empty.
            batch_size: Override the instance-level batch size for this call.
                        Defaults to the value set in ``__init__``.

        Returns:
            Float32 numpy array of shape ``(len(texts), 768)``.

        Raises:
            ValueError: When ``texts`` is empty.

        Example:
            >>> embedder = BertEmbedder()
            >>> vecs = embedder.encode_batch(["Hello", "World"])
            >>> vecs.shape
            (2, 768)
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        bs = batch_size if batch_size is not None else self._batch_size
        chunks: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            chunk = texts[i: i + bs]
            encoding = self._tokenizer(
                chunk,
                max_length=self._max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self._device)
            attention_mask = encoding["attention_mask"].to(self._device)

            with torch.no_grad():
                # SentimentClassifier.bert  → BertForSequenceClassification
                # .bert                     → BertModel (raw transformer encoder)
                outputs = self._model.bert.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
                # last_hidden_state: (batch, seq_len, 768)
                # Position 0 is always the [CLS] token.
                cls_vecs = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
                chunks.append(cls_vecs.cpu().numpy())

        return np.vstack(chunks).astype(np.float32)
