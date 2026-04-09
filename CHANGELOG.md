# Changelog

All notable changes to **Social Sentiment Tracker** are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

> **Documentation policy:** Every entry that touches user-visible behaviour, public API,
> or installation steps must include a corresponding update to both `README.md` and `README_CN.md`
> in the **same commit**.

---

## [1.4.0] — 2026-04-08

### Added
- `reports/figures/screenshot_home.png`, `screenshot_eda.png`, `screenshot_live_demo.png`, `screenshot_comparison.png` — automated headless Playwright screenshots of all four Streamlit pages
- `notebooks/03_bert_finetune.ipynb` patched and executed: 1 epoch, CPU, 240 samples (demo run); BERT metrics: Acc 0.4058 · F1 0.2844

### Changed
- `README.md` / `README_CN.md` — Screenshots section: uncommented all four page images (now live)
- `README.md` / `README_CN.md` — Model Performance table: filled BERT column with quick-run demo results with explanatory footnote
- `notebooks/03_bert_finetune.ipynb` — epochs 2 → 1; confusion matrix uses dynamic class labels; ROC curve skips gracefully for multi-class

---

## [1.3.0] — 2026-04-08

### Added
- `reports/figures/` — all visualisation PNGs committed: confusion matrix, word clouds (Negative / Positive / Neutral), sentiment distribution, text-length histogram, top-keywords chart, model-comparison bar chart, ROC curve
- `notebooks/01_eda_executed.ipynb` — EDA notebook fully executed (no errors); output excluded from git via `.gitignore` (`*_executed.ipynb`)
- `notebooks/02_baseline_model_executed.ipynb` — baseline notebook fully executed; output excluded from git

### Changed
- `.gitignore` — added `*_executed.ipynb` (cell outputs contain local absolute paths) and `run_training.py` (temporary one-off script)
- `notebooks/02_baseline_model.ipynb` — fixed confusion matrix to infer class labels dynamically; fixed ROC curve cell to skip gracefully for multi-class data and print OvR AUC instead
- `README.md` / `README_CN.md` — corrected Baseline ROC-AUC from 0.9722 → **0.9717** (actual run result)

---

## [1.2.0] — 2026-04-08

### Added
- `data/processed/train.csv`, `val.csv`, `test.csv` — persisted split files written by `split_data()`
- `split_data()` gained optional `save_dir` parameter; splits are now auto-saved to `data/processed/` on every call
- 50 unique templates per sentiment class in `generate_mock_data()` (up from 10); ensures ~340+ unique rows after deduplication

### Changed
- `generate_mock_data()` — added prefix / suffix variation logic for greater textual diversity
- `README.md` / `README_CN.md` — Model Performance table filled with real Baseline test-set results:
  Accuracy 0.8841 · Precision 0.9014 · Recall 0.8841 · F1 0.8824 · ROC-AUC 0.9717

---

## [Unreleased]

> Items planned for future releases. Move entries to a versioned section upon release.

### Planned
- Multi-class fine-grained sentiment (5-class)
- Live Twitter / Reddit API data ingestion
- Aspect-Based Sentiment Analysis (ABSA)
- SHAP / LIME model explainability
- INT8 quantised BERT inference for CPU speedup
- Docker containerisation
- GitHub Actions CI/CD (lint + test + build)
- `pytest` unit test suite for all `src/` modules

---

## [1.1.0] — 2026-04-08

### Added
- `README_CN.md` — complete Chinese translation of `README.md` (bilingual documentation policy established)
- `CHANGELOG.md` — version history following Keep a Changelog format (this file)
- `CONTRIBUTING.md` — comprehensive documentation maintenance standards including:
  - Bilingual sync policy (README.md + README_CN.md must update together)
  - Documentation update decision table (when each file must change)
  - Google-style docstring template with mandatory `Args`, `Returns`, `Raises`, `Example` sections
  - Conventional Commits format guide with scopes for each `src/` module
  - Branch naming conventions
  - Pull request checklist template

### Changed
- `README.md` — major expansion:
  - Added bilingual navigation badge (`English · 中文`)
  - Added Table of Contents
  - Added `Key Features` section (8 features with detail column)
  - Expanded architecture diagram to show all six `src/` modules with function names
  - Added `Version` column to Tech Stack table
  - Added hyperparameter reference table in Installation section
  - Added `Key Outputs` column to Notebooks table
  - Added full `API Overview` section (all 6 modules, 28 public functions documented)
  - Added `Screenshots` section with placeholders for all 4 Streamlit pages
  - Expanded `Future Roadmap` from 6 to 8 items
  - Added `Documentation Standards` section linking to `CONTRIBUTING.md`
  - Added `Changelog` section linking to `CHANGELOG.md`

---

## [1.0.1] — 2026-04-08

### Fixed
- `src/baseline_model.py` — removed deprecated `multi_class='auto'` parameter from `LogisticRegression` constructor; resolves `FutureWarning` in scikit-learn ≥ 1.5

---

## [1.0.0] — 2026-04-08

### Added

**Core pipeline modules (`src/`)**
- `src/data_loader.py`
  - `load_sentiment140(filepath)` — load Sentiment140 CSV; map label 4 → 1 for binary classification
  - `clean_text(text)` — lower-case, strip URLs / @mentions / #hashtags / non-alpha characters
  - `preprocess_dataframe(df)` — apply `clean_text`, drop empty / duplicate rows
  - `split_data(df, test_size=0.2, val_size=0.1)` — stratified train / val / test split
  - `generate_mock_data(n=500, save_path=None)` — 500-sample balanced mock CSV with date column
  - `load_data(real_path=None)` — convenience loader: try real dataset, fall back to mock

- `src/preprocess.py`
  - `tokenize(text)` — NLTK `word_tokenize` with whitespace-split fallback
  - `remove_stopwords(tokens, language='english')` — filter NLTK stopword list
  - `lemmatize(tokens)` — WordNetLemmatizer on token list
  - `add_text_features(df, text_col='clean_text')` — add `word_count`, `char_count`, `avg_word_len`, `unique_word_ratio`

- `src/baseline_model.py`
  - `build_pipeline()` — TF-IDF (max_features=50k, ngram=(1,2), sublinear_tf=True) → LogReg sklearn Pipeline
  - `train_baseline(train_df, val_df)` — fit, log val metrics, save `models/baseline_tfidf_lr.pkl`
  - `predict(pipeline, texts)` — return `(labels, probabilities)` arrays
  - `load_baseline_model(path=None)` — load serialised Pipeline via joblib

- `src/bert_model.py`
  - `SentimentDataset` — PyTorch `Dataset`; returns `input_ids`, `attention_mask`, `label` tensors
  - `SentimentClassifier(nn.Module)` — BERT encoder → Dropout(0.3) → Linear(hidden, num_labels)
  - `train_bert(train_df, val_df, config=None)` — AdamW + linear warmup; saves best val_acc checkpoint to `models/bert_sentiment.pt`
  - `predict_bert(model, tokenizer, texts, device)` — batch inference; returns `(labels, confidences)`
  - `load_bert_model(path=None, num_labels=2)` — load state-dict + tokenizer; returns `(model, tokenizer)`

- `src/evaluate.py`
  - `evaluate_model(y_true, y_pred, model_name, y_scores=None)` — accuracy, precision, recall, F1, ROC-AUC; print classification report
  - `plot_confusion_matrix(y_true, y_pred, model_name, labels=None)` — normalised seaborn heatmap; save PNG
  - `plot_roc_curve(y_true, y_scores, model_name)` — ROC curve with AUC; save PNG
  - `compare_models(baseline_results, bert_results)` — side-by-side bar chart + comparison DataFrame

- `src/visualize.py`
  - `plot_sentiment_distribution(df)` — interactive Plotly donut / pie chart
  - `plot_text_length_distribution(df)` — overlapping histogram by sentiment class
  - `plot_wordcloud(df, sentiment=1)` — save word cloud PNG; colour map varies by class
  - `plot_sentiment_over_time(df, freq='D')` — daily / weekly trend line chart
  - `plot_top_keywords(df, n=20, sentiment=None)` — TF-IDF horizontal bar chart
  - `plot_confidence_gauge(confidence, sentiment_label)` — Plotly Indicator gauge for Streamlit

**Web demo (`app/`)**
- `app/streamlit_app.py` — four-page Streamlit application:
  - Page 1 Home: dataset stats, sentiment pie, architecture overview
  - Page 2 Data Analysis: word cloud, text-length histogram, keyword chart, time-series trend
  - Page 3 Live Demo: single-text or batch input; Baseline or BERT selector; confidence gauge; batch CSV download
  - Page 4 Model Comparison: metrics table, bar chart, ROC curves, written analysis

**Configuration & infrastructure**
- `config.py` — centralised paths (`ROOT_DIR`, `RAW_DATA_DIR`, `MODELS_DIR`, `FIGURES_DIR`), hyperparameters (`BERT_MODEL_NAME="bert-base-uncased"`, `MAX_LENGTH=128`, `BATCH_SIZE=16`, `EPOCHS=3`, `LEARNING_RATE=2e-5`, `TFIDF_MAX_FEATURES=50_000`, `RANDOM_SEED=42`), `set_seed()`, `get_logger()`
- `environment.yml` — Conda env `sentiment-tracker` (Python 3.10, pytorch channel, cpuonly build)
- `requirements.txt` — pip-installable dependency list with version bounds
- `.gitignore` — ignores `models/*.pkl`, `models/*.pt`, `data/raw/`, `__pycache__/`, `.env`, `.ipynb_checkpoints/`

**Notebooks**
- `notebooks/01_eda.ipynb` — data loading, missing-value analysis, class distribution, text-length statistics, high-frequency word analysis, TF-IDF keywords, temporal trends, split summary
- `notebooks/02_baseline_model.ipynb` — train TF-IDF + LR pipeline, test-set evaluation, confusion matrix, ROC curve, LR coefficient feature importance
- `notebooks/03_bert_finetune.ipynb` — BERT fine-tuning, per-epoch training curves, test-set evaluation, baseline comparison, error analysis

**Documentation**
- `README.md` — initial English documentation with architecture, tech stack, installation, project structure, usage, model performance table, notebooks overview, future roadmap

---

*All dates use `YYYY-MM-DD` format (ISO 8601).*
