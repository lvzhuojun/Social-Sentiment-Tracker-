"""
src/bert_model.py — BERT fine-tuning for sentiment classification.

Uses HuggingFace Transformers (bert-base-uncased) with a custom classification
head. Training is compatible with both CUDA-enabled GPUs and CPU-only machines.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    BATCH_SIZE,
    BERT_MODEL_NAME,
    BERT_MODEL_PATH,
    EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE,
    MAX_LENGTH,
    RANDOM_SEED,
    WARMUP_RATIO,
    get_logger,
    set_seed,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch / transformers imports (fail gracefully in CPU-only envs)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        BertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch / Transformers not available. BERT training disabled.")


def _check_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch and HuggingFace Transformers are required for BERT training. "
            "Install them with: pip install torch transformers"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset:
    """PyTorch Dataset wrapping tokenised sentiment examples.

    Args:
        texts: List of input strings.
        labels: List of integer labels (0, 1, …).
        tokenizer: HuggingFace tokenizer compatible with the chosen BERT model.
        max_length: Maximum token sequence length (default from config).

    Example:
        >>> dataset = SentimentDataset(texts, labels, tokenizer)
        >>> dataset[0]
        {'input_ids': tensor([...]), 'attention_mask': tensor([...]), 'label': tensor(1)}
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = MAX_LENGTH,
    ) -> None:
        _check_torch()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SentimentClassifier(nn.Module):
    """BERT-based binary / multi-class sentiment classifier.

    Architecture:
        BERT encoder → Dropout(0.3) → Linear(hidden_size, num_labels)

    Args:
        num_labels: Number of output classes (default 2 for binary).
        model_name: HuggingFace model identifier (default ``bert-base-uncased``).
        dropout: Dropout probability applied before the classification head.

    Example:
        >>> model = SentimentClassifier(num_labels=2)
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = BERT_MODEL_NAME,
        dropout: float = 0.3,
    ) -> None:
        _check_torch()
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        labels: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Forward pass.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.
            attention_mask: Attention mask tensor of shape ``(batch, seq_len)``.
            labels: Optional ground-truth labels for loss computation.

        Returns:
            HuggingFace ``SequenceClassifierOutput`` object.
            Access ``.logits`` (and ``.loss`` when labels are provided).
        """
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict | None = None,
    text_col: str = "clean_text",
    label_col: str = "label",
) -> Tuple:
    """Fine-tune BERT on the training set with early-stopping on val accuracy.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        config: Optional dict to override training hyperparameters.
                Supported keys: ``epochs``, ``batch_size``, ``learning_rate``,
                ``max_length``, ``model_name``.
        text_col: Text column name (default ``'clean_text'``).
        label_col: Label column name (default ``'label'``).

    Returns:
        Tuple ``(model, tokenizer)`` — the best checkpoint loaded in eval mode.

    Side-effects:
        * Logs per-epoch train_loss / val_loss / val_accuracy.
        * Saves best model weights to ``config.BERT_MODEL_PATH``.

    Example:
        >>> model, tokenizer = train_bert(train_df, val_df)
    """
    _check_torch()
    set_seed(RANDOM_SEED)

    # --- resolve config ---
    cfg = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "model_name": BERT_MODEL_NAME,
    }
    if config:
        cfg.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # --- tokenizer ---
    logger.info("Loading tokenizer: %s", cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # --- determine number of unique labels ---
    unique_labels = sorted(train_df[label_col].unique())
    num_labels = len(unique_labels)
    logger.info("Number of labels: %d  (%s)", num_labels, unique_labels)

    # --- datasets & loaders ---
    train_dataset = SentimentDataset(
        train_df[text_col].tolist(),
        train_df[label_col].tolist(),
        tokenizer,
        max_length=cfg["max_length"],
    )
    val_dataset = SentimentDataset(
        val_df[text_col].tolist(),
        val_df[label_col].tolist(),
        tokenizer,
        max_length=cfg["max_length"],
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    # --- model ---
    model = SentimentClassifier(num_labels=num_labels, model_name=cfg["model_name"])
    model.to(device)

    # --- optimiser & scheduler ---
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    BERT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                output = model(input_ids, attention_mask, labels=labels)
                val_loss += output.loss.item()
                preds = output.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        logger.info(
            "Epoch %d/%d — train_loss: %.4f | val_loss: %.4f | val_acc: %.4f",
            epoch, cfg["epochs"], avg_train_loss, avg_val_loss, val_acc,
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BERT_MODEL_PATH)
            logger.info("  ↑ New best val_acc %.4f — model saved.", best_val_acc)

    # Reload best weights
    model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    logger.info("Training complete. Best val_acc: %.4f", best_val_acc)

    # Persist val metrics to reports/metrics.json for the Streamlit app
    try:
        import json
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["label"].to(device)
                out = model(ids, mask)
                preds = out.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_true.extend(lbls.cpu().numpy().tolist())

        import numpy as np
        yt = np.array(all_true)
        yp = np.array(all_preds)
        metrics = {
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "precision": round(
                float(precision_score(yt, yp, average="weighted", zero_division=0)), 4),
            "recall": round(
                float(recall_score(yt, yp, average="weighted", zero_division=0)), 4),
            "f1": round(
                float(f1_score(yt, yp, average="weighted", zero_division=0)), 4),
            "roc_auc": None,
        }
        metrics_path = FIGURES_DIR.parent / "metrics.json"
        existing: dict = {}
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as fh:
                existing = json.load(fh)
        existing["bert"] = metrics
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)
        logger.info("BERT metrics saved to %s", metrics_path)
    except Exception as exc:
        logger.warning("Could not save BERT metrics JSON: %s", exc)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_bert(
    model,
    tokenizer,
    texts: List[str],
    device: "torch.device | None" = None,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-predict sentiment labels and confidence scores with a BERT model.

    Args:
        model: Fine-tuned :class:`SentimentClassifier` in eval mode.
        tokenizer: Matching HuggingFace tokenizer.
        texts: List of input strings.
        device: Torch device. Auto-detected if ``None``.
        batch_size: Inference batch size (default from config).
        max_length: Maximum sequence length (default from config).

    Returns:
        Tuple ``(labels, confidences)`` where:
        * ``labels`` — 1-D int array of predicted class indices.
        * ``confidences`` — 1-D float array of max softmax probability.

    Example:
        >>> labels, confs = predict_bert(model, tokenizer, ["Great product!"])
        >>> labels
        array([1])
    """
    _check_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_labels: List[int] = []
    all_confs: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            preds = probs.argmax(dim=-1)

        all_labels.extend(preds.cpu().numpy().tolist())
        all_confs.extend(probs.max(dim=-1).values.cpu().numpy().tolist())

    return np.array(all_labels, dtype=int), np.array(all_confs, dtype=float)


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_bert_model(
    path: Path | None = None,
    num_labels: int = 2,
    model_name: str = BERT_MODEL_NAME,
) -> Tuple:
    """Load a saved BERT model and its tokenizer.

    Args:
        path: Path to the ``.pt`` state-dict file. Defaults to
              ``config.BERT_MODEL_PATH``.
        num_labels: Number of output classes (must match the saved model).
        model_name: HuggingFace model identifier.

    Returns:
        Tuple ``(model, tokenizer)`` ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.

    Example:
        >>> model, tokenizer = load_bert_model()
    """
    _check_torch()
    model_path = Path(path) if path else BERT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"BERT model not found at {model_path}. Run train_bert() first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentimentClassifier(num_labels=num_labels, model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    logger.info("BERT model loaded from %s (device=%s)", model_path, device)
    return model, tokenizer
