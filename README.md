# Social Sentiment Tracker

![CI](https://github.com/lvzhuojun/social-sentiment-tracker/actions/workflows/ci.yml/badge.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://social-sentiment-tracker.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)

> An end-to-end NLP platform that ingests raw social-media text, cleans it, trains two complementary sentiment models — a fast TF-IDF + Logistic Regression baseline and a fine-tuned **BERT** (`bert-base-uncased`) — and serves predictions through an interactive four-page **Streamlit** web demo and a **FastAPI** REST endpoint. Trained on the **TweetEval** benchmark dataset (59,899 real tweets, 3-class sentiment) with an automatic mock-data fallback for offline development.

![Streamlit Demo](reports/figures/screenshot_home.png)

**Language / 语言:** [English](#) · [中文](README_CN.md)

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Notebooks](#notebooks)
- [API Overview](#api-overview)
- [Screenshots](#screenshots)
- [Future Roadmap](#future-roadmap)
- [Documentation Standards](#documentation-standards)
- [Changelog](#changelog)
- [Author](#author)

---

## Key Features

| # | Feature | Detail |
|---|---------|--------|
| 1 | **Dual-Model Pipeline** | TF-IDF + LogReg baseline (CPU, seconds) vs. fine-tuned BERT (GPU/CPU, ~1.5 hr on RTX 5060) |
| 2 | **TweetEval Benchmark Data** | 59,899 real tweets, 3-class labels; `download_data.py` fetches from HuggingFace automatically |
| 3 | **FastAPI REST Endpoint** | `api/serve.py` — `/predict`, `/predict/batch`, `/health`; Pydantic validation; uvicorn-ready |
| 4 | **SHAP Explainability** | `src/explain.py` — SHAP `LinearExplainer` for baseline; token-level attribution in Streamlit |
| 5 | **Hyperparameter Tuning** | `scripts/tune_baseline.py` — 3-fold GridSearchCV across TF-IDF and LR params with heatmap |
| 6 | **Error Analysis Notebook** | `04_error_analysis.ipynb` — high-confidence errors, negation impact, class difficulty, SHAP on errors |
| 7 | **Reproducible Experiments** | `set_seed(42)` fixes `random`, `numpy`, and `torch` seeds globally from `config.py` |
| 8 | **Full Engineering Stack** | 121 pytest tests · GitHub Actions CI · Docker · Streamlit Cloud · Google-style docstrings |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Raw Text Input                                │
│             Sentiment140 CSV  /  Auto-generated mock data             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   data_loader.py    │
                    │  load_sentiment140()│
                    │  clean_text()       │
                    │  preprocess_df()    │
                    │  split_data()       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
   ┌──────────▼──────────┐           ┌──────────▼──────────┐
   │   preprocess.py     │           │   preprocess.py     │
   │   tokenize()        │           │   add_text_features │
   │   remove_stopwords()│           │   word_count        │
   │   lemmatize()       │           │   char_count        │
   └──────────┬──────────┘           └──────────┬──────────┘
              │                                 │
   ┌──────────▼──────────┐           ┌──────────▼──────────┐
   │  baseline_model.py  │           │    bert_model.py    │
   │  TfidfVectorizer    │           │  SentimentDataset   │
   │  LogisticRegression │           │  SentimentClassifier│
   │  build_pipeline()   │           │  bert-base-uncased  │
   │  train_baseline()   │           │  Dropout(0.3)       │
   │  predict()          │           │  Linear Head        │
   └──────────┬──────────┘           │  train_bert()       │
              │                      │  predict_bert()     │
              │                      └──────────┬──────────┘
              └──────────────┬──────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │     evaluate.py     │
                  │  evaluate_model()   │
                  │  confusion_matrix() │
                  │  plot_roc_curve()   │
                  │  compare_models()   │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │    visualize.py     │
                  │  sentiment_dist()   │
                  │  text_length_dist() │
                  │  plot_wordcloud()   │
                  │  sentiment_time()   │
                  │  top_keywords()     │
                  │  confidence_gauge() │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ app/streamlit_app   │
                  │  Page 1 · Home      │
                  │  Page 2 · EDA       │
                  │  Page 3 · Live Demo │
                  │  Page 4 · Compare   │
                  └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10 |
| Data | TweetEval (`tweet_eval/sentiment`, HuggingFace) / Mock CSV | 59,899 tweets |
| ML Baseline | scikit-learn · TF-IDF + LogReg | ≥ 1.3.0 |
| Deep Learning | PyTorch (CUDA 12.8) | ≥ 2.1.0 |
| NLP Transformers | HuggingFace `transformers` (`bert-base-uncased`) | ≥ 4.35.0 |
| NLP Helpers | NLTK (tokenize · stopwords · lemmatize) | ≥ 3.8.1 |
| Explainability | SHAP (`LinearExplainer`) | ≥ 0.44.0 |
| REST API | FastAPI + uvicorn | ≥ 0.110.0 |
| Interactive Viz | Plotly | ≥ 5.18.0 |
| Static Viz | Matplotlib · Seaborn | ≥ 3.7.0 / 0.12.0 |
| Word Cloud | wordcloud | ≥ 1.9.2 |
| Web Demo | Streamlit | ≥ 1.28.0 |
| Model Serialisation | joblib | ≥ 1.3.0 |
| Environment | Conda (`sentiment-tracker`) | — |

---

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

### Clone and Setup

```bash
# 1. Clone the repository
git clone https://github.com/lvzhuojun/social-sentiment-tracker.git
cd social-sentiment-tracker

# 2. Create and activate the conda environment (Python 3.10 + all dependencies)
conda env create -f environment.yml
conda activate sentiment-tracker

# 3. Download the TweetEval dataset (59,899 tweets from HuggingFace)
python scripts/download_data.py
#    If absent, 500 balanced mock samples are auto-generated on first run.

# 4. Train models
python scripts/train_full.py --model baseline  # TF-IDF + LR (~7 s)
python scripts/train_full.py --model bert      # BERT fine-tuning (~90 min GPU)

# 5. Launch the Streamlit web demo
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### Streamlit Cloud (Zero-Install Public Demo)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork, branch `main`, file `app/streamlit_app.py`
4. Set **Python version** to `3.10` and **Requirements file** to `requirements-cloud.txt`
5. Click **Deploy** — the baseline model auto-trains on first start (~10 s)

> BERT inference is disabled on the free tier (memory limit). All other pages work fully.

---

### Docker (One-Command Demo)

```bash
# Build the image (first build ~5 min due to torch download)
docker build -t social-sentiment-tracker .

# Run the Streamlit demo
docker run -p 8501:8501 social-sentiment-tracker
# Opens at http://localhost:8501
```

To use a pre-trained model inside the container, mount the `models/` directory:

```bash
docker run -p 8501:8501 \
  -v "$(pwd)/models:/app/models" \
  social-sentiment-tracker
```

---

### GPU Support (Optional)

Open `environment.yml` and make the following change:

```yaml
# Remove this line:
- cpuonly
# Add this line (match your CUDA version):
- pytorch-cuda=12.1
```

Then recreate the environment:

```bash
conda env remove -n sentiment-tracker
conda env create -f environment.yml
```

All training hyperparameters are controlled in `config.py`:

| Parameter | Default | Config Key |
|-----------|---------|------------|
| BERT model | `bert-base-uncased` | `BERT_MODEL_NAME` |
| Max sequence length | 128 | `MAX_LENGTH` |
| Batch size | 16 | `BATCH_SIZE` |
| Epochs | 3 | `EPOCHS` |
| Learning rate | 2e-5 | `LEARNING_RATE` |
| Warmup ratio | 0.1 | `WARMUP_RATIO` |
| TF-IDF max features | 50,000 | `TFIDF_MAX_FEATURES` |
| Random seed | 42 | `RANDOM_SEED` |

---

## Project Structure

```
social-sentiment-tracker/
│
├── app/
│   └── streamlit_app.py        # Four-page web demo (Home · EDA · Live Demo · Comparison)
│
├── data/
│   ├── raw/                    # Original CSVs — git-ignored; mock_data.csv auto-generated
│   └── processed/              # Cleaned & split DataFrames (populated at runtime)
│
├── models/                     # Saved model artefacts — git-ignored
│   ├── baseline_tfidf_lr.pkl   # Serialised sklearn Pipeline (joblib)
│   └── bert_sentiment.pt       # BERT state-dict checkpoint (best val_acc)
│
├── api/
│   ├── __init__.py
│   ├── serve.py                # FastAPI: /predict · /predict/batch · /health
│   └── requirements.txt        # fastapi · uvicorn · pydantic
│
├── notebooks/
│   ├── 01_eda.ipynb            # EDA: class distribution, text stats, word clouds
│   ├── 02_baseline_model.ipynb # Train TF-IDF + LR, feature importance, evaluation
│   ├── 03_bert_finetune.ipynb  # Fine-tune BERT, training curves
│   └── 04_error_analysis.ipynb # Error analysis: high-confidence failures, negation, SHAP
│
├── reports/
│   └── figures/                # Auto-saved PNGs: confusion matrices, ROC, tuning heatmap
│
├── scripts/
│   ├── download_data.py        # Download TweetEval from HuggingFace
│   ├── train_full.py           # End-to-end training (--model baseline|bert|both)
│   └── tune_baseline.py        # GridSearchCV hyperparameter search + heatmap
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # load_tweet_eval · clean_text · split_data · generate_mock_data
│   ├── preprocess.py           # tokenize · remove_stopwords · lemmatize · add_text_features
│   ├── baseline_model.py       # build_pipeline · train_baseline · predict · load_baseline_model
│   ├── bert_model.py           # SentimentDataset · SentimentClassifier · train_bert · predict_bert
│   ├── evaluate.py             # evaluate_model · plot_confusion_matrix · plot_roc_curve (OvR) · compare_models
│   ├── visualize.py            # plot_sentiment_distribution · plot_wordcloud · plot_top_keywords · …
│   └── explain.py              # explain_baseline_prediction (SHAP) · shap_to_plotly_bar
│
├── config.py                   # Centralised paths, hyperparameters, set_seed(), get_logger()
├── environment.yml             # Conda environment spec (Python 3.10, pytorch channel)
├── requirements.txt            # pip-installable dependencies with version bounds
├── Dockerfile                  # python:3.10-slim image, port 8501, healthcheck
│
├── README.md                   # English documentation (this file)
├── README_CN.md                # Chinese documentation (中文文档)
├── CHANGELOG.md                # Version history and release notes
├── CONTRIBUTING.md             # Contribution and documentation maintenance standards
└── UPDATE_RULES.md             # Mandatory standards synchronisation checklist
```

---

## Usage

### Train Models

```bash
conda activate sentiment-tracker

# ── Download real data first (59,899 TweetEval tweets) ───────────────────
python scripts/download_data.py

# ── Baseline (TF-IDF + Logistic Regression) ──────────────────────────────
# Trains in ~7 seconds on CPU.  Output: models/baseline_tfidf_lr.pkl
python scripts/train_full.py --model baseline

# ── BERT Fine-tuning ──────────────────────────────────────────────────────
# GPU strongly recommended (≥ 6 GB VRAM, ~90 min on RTX 5060).
# Falls back to CPU (~8–12 hr per epoch). Output: models/bert_sentiment.pt
python scripts/train_full.py --model bert

# ── Both models in sequence ───────────────────────────────────────────────
python scripts/train_full.py

# ── Hyperparameter grid search (baseline only, ~15 min) ──────────────────
python scripts/tune_baseline.py
```

### Start the FastAPI Server

```bash
conda activate sentiment-tracker
pip install -r api/requirements.txt
uvicorn api.serve:app --host 0.0.0.0 --port 8000

# Single prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love this product!"}'

# Health check
curl http://localhost:8000/health

# Interactive API docs
open http://localhost:8000/docs
```

### Run the Web Demo

```bash
conda activate sentiment-tracker
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

**Page descriptions:**

| Page | Description |
|------|-------------|
| **Home** | Dataset stats (total samples, class counts), sentiment distribution pie chart, architecture overview |
| **Data Analysis** | Text-length histogram, word cloud (per sentiment), TF-IDF keyword bar chart, sentiment trend over time |
| **Live Demo** | Single-text or batch mode; choose Baseline or BERT; outputs sentiment label + confidence gauge; batch results downloadable as CSV |
| **Model Comparison** | Accuracy / Precision / Recall / F1 / AUC table, bar chart, and ROC curves side-by-side |

### Run Notebooks

```bash
conda activate sentiment-tracker
jupyter lab
# Open notebooks/ in order: 01 → 02 → 03
```

---

## Model Performance

Results on the **TweetEval held-out test set** (12,264 samples, `RANDOM_SEED=42`).
Three-class sentiment: 0 = Negative · 1 = Positive · 2 = Neutral.

| Metric | Baseline (TF-IDF + LR) | BERT (bert-base-uncased) |
|--------|------------------------|--------------------------|
| Accuracy | 0.5935 | **0.6889** |
| Precision (weighted) | 0.6096 | **0.6973** |
| Recall (weighted) | 0.5935 | **0.6889** |
| F1 (weighted) | 0.5788 | **0.6873** |
| ROC-AUC (macro OvR) | 0.7724 | **0.8660** |

> Dataset: `tweet_eval/sentiment` from HuggingFace (45,615 train / 2,000 val / 12,284 test).
> BERT fine-tuned for 3 epochs on RTX 5060 Laptop GPU (~28.5 min); best checkpoint at Epoch 2
> (val_acc 0.7175). All metrics computed by `src/evaluate.evaluate_model()`; confusion matrices
> and ROC curves saved to `reports/figures/` with OvR multi-class ROC support.

**Per-class breakdown — Baseline:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative (0) | 0.68 | 0.35 | 0.46 | 3,968 |
| Positive (1) | 0.53 | 0.62 | 0.57 | 2,371 |
| Neutral (2) | 0.59 | 0.75 | 0.66 | 5,925 |

**Per-class breakdown — BERT:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative (0) | 0.67 | 0.79 | 0.72 | 3,968 |
| Positive (1) | 0.62 | 0.73 | 0.67 | 2,371 |
| Neutral (2) | 0.75 | 0.61 | 0.67 | 5,925 |

> **BERT improves over baseline by +9.5 pp Accuracy, +10.9 pp F1, +9.4 pp AUC.**
> Negative class recall rises dramatically from 0.35 → 0.79 thanks to contextual representations.
> See `notebooks/04_error_analysis.ipynb` for a detailed failure-mode investigation.

---

## Notebooks

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_eda.ipynb` | Data loading, class distribution, text statistics, word clouds, sentiment trend over time | Plotly charts · `reports/figures/wordcloud_*.png` |
| `02_baseline_model.ipynb` | Train TF-IDF + LR, evaluate on test set, plot LR coefficients, confusion matrix, ROC | `models/baseline_tfidf_lr.pkl` · ROC curve PNG |
| `03_bert_finetune.ipynb` | Fine-tune BERT, per-epoch training curves, compare with baseline | `models/bert_sentiment.pt` · model comparison chart |
| `04_error_analysis.ipynb` | High-confidence error analysis, negation impact, class difficulty, SHAP on error cases | `reports/figures/error_*.png` · insight summary |

---

## API Overview

### `config.py`

| Symbol | Type | Description |
|--------|------|-------------|
| `set_seed(seed=42)` | function | Fix `random` / `numpy` / `torch` seeds for reproducibility |
| `get_logger(name)` | function | Return a consistently-formatted `logging.Logger` |
| `BERT_MODEL_NAME` | constant | `"bert-base-uncased"` |
| `MAX_LENGTH` | constant | Max token sequence length (128) |
| `BATCH_SIZE` | constant | Training batch size (16) |
| `EPOCHS` | constant | Fine-tuning epochs (3) |
| `LEARNING_RATE` | constant | AdamW learning rate (2e-5) |

---

### `src/data_loader.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_data` | `load_data(real_path=None)` | Auto-select Sentiment140 or mock fallback; returns preprocessed DataFrame |
| `load_sentiment140` | `load_sentiment140(filepath)` | Load raw CSV; map label 4 → 1 for binary classification |
| `clean_text` | `clean_text(text: str) -> str` | Lower-case; strip URLs, @mentions, #hashtags, non-alpha characters |
| `preprocess_dataframe` | `preprocess_dataframe(df)` | Apply `clean_text`, drop empty / duplicate rows |
| `split_data` | `split_data(df, test_size=0.2, val_size=0.1)` | Stratified train / val / test split; returns three DataFrames |
| `generate_mock_data` | `generate_mock_data(n=500, save_path=None)` | Generate balanced synthetic sentiment CSV with date column |

---

### `src/preprocess.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `tokenize` | `tokenize(text: str) -> List[str]` | NLTK `word_tokenize` with whitespace-split fallback |
| `remove_stopwords` | `remove_stopwords(tokens, language='english')` | Filter NLTK English stopword list |
| `lemmatize` | `lemmatize(tokens: List[str]) -> List[str]` | WordNetLemmatizer on token list |
| `add_text_features` | `add_text_features(df, text_col='clean_text')` | Add `word_count`, `char_count`, `avg_word_len`, `unique_word_ratio` columns |

---

### `src/baseline_model.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_pipeline` | `build_pipeline() -> Pipeline` | Construct TF-IDF (max_features=50k, ngram=(1,2)) → LogReg sklearn Pipeline |
| `train_baseline` | `train_baseline(train_df, val_df) -> Pipeline` | Fit pipeline, log val metrics, save to `models/baseline_tfidf_lr.pkl` |
| `predict` | `predict(pipeline, texts) -> (labels, probs)` | Return predicted labels and probability matrix |
| `load_baseline_model` | `load_baseline_model(path=None) -> Pipeline` | Load serialised Pipeline from disk via joblib |

---

### `src/bert_model.py`

| Class / Function | Description |
|------------------|-------------|
| `SentimentDataset` | PyTorch `Dataset`; tokenises text with HuggingFace tokenizer; returns `input_ids`, `attention_mask`, `label` tensors |
| `SentimentClassifier(nn.Module)` | BERT encoder → Dropout(0.3) → `Linear(hidden_size, num_labels)` classification head |
| `train_bert(train_df, val_df, config=None)` | Fine-tune with AdamW + linear warmup; saves best-val-acc checkpoint + companion `.json` config |
| `predict_bert(model, tokenizer, texts, device)` | Batch inference; returns `(labels, probs_matrix)` where `probs_matrix` is shape `(n, n_classes)` |
| `load_bert_model(path=None, num_labels=3)` | Auto-reads `num_labels` from companion `.json`; returns `(model, tokenizer)` ready for inference |

### `api/serve.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Liveness check — reports which models are loaded |
| `/predict` | `POST` | Single-text prediction: `{"text": "...", "model": "baseline"}` → `{sentiment, label, confidence, latency_ms}` |
| `/predict/batch` | `POST` | Batch prediction: `{"texts": [...], "model": "baseline"}` — up to 128 texts per request |

---

### `src/evaluate.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `evaluate_model` | `evaluate_model(y_true, y_pred, model_name, y_scores=None)` | Compute accuracy, precision, recall, F1, ROC-AUC; print classification report |
| `plot_confusion_matrix` | `plot_confusion_matrix(y_true, y_pred, model_name, labels=None)` | Normalised seaborn heatmap; save PNG to `reports/figures/` |
| `plot_roc_curve` | `plot_roc_curve(y_true, y_scores, model_name)` | ROC curve with AUC annotation; save PNG |
| `compare_models` | `compare_models(baseline_results, bert_results)` | Side-by-side bar chart + comparison DataFrame |

---

### `src/visualize.py`

| Function | Description |
|----------|-------------|
| `plot_sentiment_distribution(df)` | Interactive Plotly donut / pie chart of class counts |
| `plot_text_length_distribution(df)` | Overlapping word-count histogram coloured by sentiment class |
| `plot_wordcloud(df, sentiment=1)` | Save word cloud PNG; colour map varies by sentiment (green/red/blue) |
| `plot_sentiment_over_time(df, freq='D')` | Daily / weekly sentiment trend line chart |
| `plot_top_keywords(df, n=20, sentiment=None)` | Horizontal bar chart of top TF-IDF terms (optionally per sentiment) |
| `plot_confidence_gauge(confidence, sentiment_label)` | Circular Plotly Indicator gauge for Streamlit Live Demo page |

---

### `src/explain.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `explain_baseline_prediction` | `explain_baseline_prediction(pipeline, text, n_top=12)` | SHAP `LinearExplainer` on TF-IDF+LR pipeline; returns `(contributions, predicted_class, classes)` where contributions are `(token, shap_value)` tuples sorted by `abs(shap_value)` descending |
| `shap_to_plotly_bar` | `shap_to_plotly_bar(contributions, predicted_class, label_names=None)` | Render SHAP token attributions as a horizontal Plotly bar chart (green = pushes toward class, red = pushes away) |

---

## Screenshots

### Home Page
![Home Page](reports/figures/screenshot_home.png)
*Dataset statistics, sentiment distribution pie chart, and project architecture overview.*

### Data Analysis Page
![EDA Page](reports/figures/screenshot_eda.png)
*Interactive word clouds, text-length histograms, and TF-IDF keyword bar charts.*

### Live Demo — Single Prediction
![Live Demo](reports/figures/screenshot_live_demo.png)
*Real-time prediction with confidence gauge. Supports single text and batch CSV download mode.*

### Model Comparison Page
![Model Comparison](reports/figures/screenshot_comparison.png)
*Side-by-side metrics table, performance bar chart, and ROC curves for Baseline vs. BERT.*

---

## Future Roadmap

- [x] **Model explainability** — SHAP `LinearExplainer` for baseline (`src/explain.py`)
- [x] **Docker deployment** — `python:3.10-slim` image, port 8501, healthcheck
- [x] **CI/CD pipeline** — GitHub Actions: flake8 lint + 121 pytest tests
- [x] **FastAPI REST endpoint** — `/predict`, `/predict/batch`, `/health` (`api/serve.py`)
- [x] **Hyperparameter tuning** — 3-fold GridSearchCV with heatmap (`scripts/tune_baseline.py`)
- [x] **Error analysis** — high-confidence errors, negation patterns, per-class difficulty (`04_error_analysis.ipynb`)
- [ ] **Quantised inference** — INT8 BERT via ONNX for 3–4× CPU speedup
- [ ] **Multi-class sentiment** — 5-class fine-grained labels (very negative → very positive)
- [ ] **Live data ingestion** — Twitter / Reddit API streaming pipeline
- [ ] **Aspect-Based Sentiment Analysis (ABSA)** — entity-level opinion mining
- [ ] **MLflow experiment tracking** — log params, metrics, and model artefacts per run

---

## Documentation Standards

This repository maintains **bilingual documentation**. `README.md` (English) and [`README_CN.md`](README_CN.md) (Chinese) must be updated **in the same commit** for every user-visible change — including API changes, new features, installation steps, and project structure updates.

All public functions follow **Google-style docstrings** with `Args`, `Returns`, `Raises`, and `Example` blocks. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full documentation maintenance policy, docstring template, commit format guide, and PR checklist.

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the complete version history.

---

## Author

**Zhuojun Lyu (吕卓俊)**
[GitHub](https://github.com/lvzhuojun) · [LinkedIn](https://www.linkedin.com/in/zhuojun-lyu/) · [Email](mailto:lzj2729033776@gmail.com)

---

*Built with Python 3.10 · HuggingFace Transformers · PyTorch · scikit-learn · Streamlit*
