# Social Sentiment Tracker

> A multi-source NLP platform for real-time social sentiment analysis — built as a portfolio project to demonstrate a complete machine-learning pipeline from raw text to production-ready predictions.

<!-- Demo screenshot placeholder -->
<!-- ![Demo](reports/figures/demo_screenshot.png) -->

---

## Motivation

Social platforms generate millions of text posts every day. Understanding the collective sentiment behind these posts — product reviews, news reactions, public discourse — is valuable for businesses, researchers, and policy-makers. This project builds an end-to-end pipeline that:

1. **Ingests** raw social text (Twitter / mock data)
2. **Cleans and preprocesses** it with NLP best practices
3. **Trains two models** — a fast TF-IDF baseline and a fine-tuned BERT model
4. **Compares and evaluates** both on held-out data
5. **Serves predictions** through an interactive Streamlit web demo

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Raw Text Input                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    clean_text()
                           │
          ┌────────────────┴────────────────┐
          │                                 │
   TF-IDF Vectorizer                BERT Tokenizer
          │                                 │
  Logistic Regression          bert-base-uncased
          │                      + Linear Head
          │                                 │
          └────────────────┬────────────────┘
                           │
                   Sentiment Label
              (Positive / Negative / Neutral)
                   + Confidence Score
                           │
                    Streamlit Demo
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Data | Sentiment140 (Twitter) / mock data |
| ML Baseline | scikit-learn (TF-IDF + LogReg) |
| Deep Model | HuggingFace Transformers (BERT) |
| Training | PyTorch 2.x |
| Visualisation | Plotly, Matplotlib, Seaborn, WordCloud |
| Web Demo | Streamlit |
| Environment | Conda |

---

## Installation

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:lvzhuojun/Social-Sentiment-Tracker-.git
cd Social-Sentiment-Tracker-

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate sentiment-tracker

# 3. (Optional) Add dataset — place twitter_training.csv in data/raw/
#    If absent, mock data is auto-generated.

# 4. Launch the web demo
streamlit run app/streamlit_app.py
```

### GPU Support (optional)
Edit `environment.yml` — replace `cpuonly` with `pytorch-cuda=12.1` (or your CUDA version).

---

## Project Structure

```
Social-Sentiment-Tracker-/
├── data/
│   ├── raw/                  # Original datasets (git-ignored)
│   └── processed/            # Cleaned, split data
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_baseline_model.ipynb
│   └── 03_bert_finetune.ipynb
├── src/
│   ├── data_loader.py        # Loading, cleaning, splitting
│   ├── preprocess.py         # Tokenisation, stopwords, lemmatisation
│   ├── baseline_model.py     # TF-IDF + Logistic Regression
│   ├── bert_model.py         # BERT fine-tuning
│   ├── evaluate.py           # Metrics, confusion matrix, ROC
│   └── visualize.py          # Plotly / WordCloud charts
├── app/
│   └── streamlit_app.py      # Four-page web demo
├── models/                   # Saved model artefacts (git-ignored)
├── reports/figures/          # Output charts
├── config.py                 # Global paths and hyperparameters
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Training the Models

```bash
# Activate environment first
conda activate sentiment-tracker

# Train baseline (fast — seconds on CPU)
python src/baseline_model.py

# Train BERT (slow on CPU — GPU recommended)
python src/bert_model.py
```

Model files are saved to `models/` and are git-ignored.

---

## Model Performance

Results on the held-out test set (fill in after training):

| Metric | Baseline (TF-IDF + LR) | BERT Fine-tuned |
|--------|------------------------|-----------------|
| Accuracy | — | — |
| Precision | — | — |
| Recall | — | — |
| F1 | — | — |
| ROC-AUC | — | — |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Data loading, class distribution, text statistics, word clouds |
| `02_baseline_model.ipynb` | Train TF-IDF + LR, evaluate, plot feature importance |
| `03_bert_finetune.ipynb` | Fine-tune BERT, compare with baseline, error analysis |

---

## Future Work

- [ ] Add multi-class support (fine-grained: very positive / positive / neutral / negative / very negative)
- [ ] Integrate live Twitter / Reddit API data collection
- [ ] Implement aspect-based sentiment analysis (ABSA)
- [ ] Add model explainability (SHAP / LIME)
- [ ] Containerise with Docker for one-command deployment
- [ ] CI/CD pipeline with GitHub Actions

---

## Author

<!-- Replace with your information -->
**[Your Name]**
[GitHub](https://github.com/lvzhuojun) · [LinkedIn](#) · [Email](#)

---

*Built with Python 3.10 · HuggingFace Transformers · scikit-learn · Streamlit*
