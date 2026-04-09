"""
app/streamlit_app.py — Social Sentiment Tracker Web Demo.

Four-page Streamlit application:
  1. Home           — project overview and dataset stats
  2. Data Analysis  — EDA charts
  3. Live Demo      — single / batch prediction with gauge
  4. Model Comparison — metrics table and bar chart
"""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup so src/ and project root are importable
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from config import FIGURES_DIR, set_seed

set_seed()

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Sentiment Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Social Sentiment Tracker")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Data Analysis", "🤖 Live Demo", "📈 Model Comparison"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · HuggingFace · scikit-learn")


# ---------------------------------------------------------------------------
# Shared data loader (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def get_data() -> pd.DataFrame:
    """Load (or generate) the sentiment dataset."""
    from src.data_loader import load_data
    return load_data()


@st.cache_resource(show_spinner="Loading baseline model…")
def get_baseline_pipeline():
    """Load or auto-train the TF-IDF + LR pipeline.

    On first run (e.g. Streamlit Cloud), the model is trained automatically
    on mock data if no saved checkpoint is found (~10 s on CPU).
    """
    try:
        from src.baseline_model import load_baseline_model
        return load_baseline_model()
    except FileNotFoundError:
        with st.spinner("First run — training baseline model on mock data (~10 s)…"):
            from src.baseline_model import train_baseline
            from src.data_loader import load_data, split_data
            df = load_data()
            train_df, val_df, _ = split_data(df)
            return train_baseline(train_df, val_df)


@st.cache_resource(show_spinner="Loading BERT model…")
def get_bert_model():
    """Load the persisted BERT model and tokenizer."""
    try:
        from src.bert_model import load_bert_model
        return load_bert_model()
    except (FileNotFoundError, RuntimeError):
        return None, None


# ===========================================================================
# PAGE 1 — HOME
# ===========================================================================
if page == "🏠 Home":
    st.title("📊 Social Sentiment Tracker")
    st.markdown(
        """
        > **A multi-source NLP platform for real-time social sentiment analysis.**

        This project demonstrates a complete machine-learning pipeline — from raw
        social media text to production-ready predictions — using two complementary
        model architectures:

        | Component | Technology |
        |-----------|-----------|
        | Data       | Sentiment140 / mock Twitter data |
        | Baseline   | TF-IDF + Logistic Regression |
        | Deep Model | BERT (bert-base-uncased) fine-tuned |
        | Frontend   | Streamlit + Plotly |
        | Serving    | scikit-learn Pipeline / HuggingFace |
        """
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    try:
        df = get_data()
        col1.metric("Total Samples", f"{len(df):,}")
        label_counts = df["label"].value_counts()
        pos = label_counts.get(1, 0)
        neg = label_counts.get(0, 0)
        col2.metric("Positive", f"{pos:,}", f"{pos/len(df)*100:.1f}%")
        col3.metric("Negative", f"{neg:,}", f"{neg/len(df)*100:.1f}%")

        st.markdown("### Dataset Distribution")
        from src.visualize import plot_sentiment_distribution
        fig = plot_sentiment_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load dataset: {exc}")

    st.markdown("---")
    st.markdown(
        """
        ### Architecture

        ```
        Raw Text ──► clean_text() ──► TF-IDF Vectorizer ──► Logistic Regression
                                  └──► BERT Tokenizer   ──► BERT + Linear Head
        ```

        ### Quick Start
        ```bash
        conda env create -f environment.yml
        conda activate sentiment-tracker
        streamlit run app/streamlit_app.py
        ```
        """
    )


# ===========================================================================
# PAGE 2 — DATA ANALYSIS
# ===========================================================================
elif page == "🔍 Data Analysis":
    st.title("🔍 Exploratory Data Analysis")

    try:
        df = get_data()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # --- Sentiment distribution ---
    st.subheader("Sentiment Distribution")
    from src.visualize import (
        plot_sentiment_distribution,
        plot_sentiment_over_time,
        plot_text_length_distribution,
        plot_top_keywords,
        plot_wordcloud,
    )
    st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)

    st.markdown("---")

    # --- Text length distribution ---
    st.subheader("Text Length Distribution")
    st.plotly_chart(plot_text_length_distribution(df), use_container_width=True)

    st.markdown("---")

    # --- Word cloud ---
    st.subheader("Word Cloud")
    wc_col1, wc_col2 = st.columns([1, 3])
    with wc_col1:
        sentiment_choice = st.selectbox(
            "Select Sentiment",
            options=[1, 0, 2],
            format_func=lambda x: {1: "Positive", 0: "Negative", 2: "Neutral"}[x],
        )
    with wc_col2:
        with st.spinner("Generating word cloud…"):
            try:
                wc_path = plot_wordcloud(df, sentiment=sentiment_choice)
                st.image(str(wc_path), use_container_width=True)
            except ImportError:
                st.info("Install wordcloud (`pip install wordcloud`) to enable this chart.")
            except Exception as exc:
                st.warning(f"Word cloud error: {exc}")

    st.markdown("---")

    # --- Top keywords ---
    st.subheader("Top TF-IDF Keywords")
    kw_col1, kw_col2 = st.columns([1, 3])
    with kw_col1:
        kw_sentiment = st.selectbox(
            "Sentiment filter",
            options=[None, 1, 0, 2],
            format_func=lambda x: "All" if x is None else
            {1: "Positive", 0: "Negative", 2: "Neutral"}[x],
        )
        n_keywords = st.slider("Number of keywords", 5, 40, 20)
    with kw_col2:
        st.plotly_chart(
            plot_top_keywords(df, n=n_keywords, sentiment=kw_sentiment),
            use_container_width=True,
        )

    st.markdown("---")

    # --- Sentiment over time ---
    if "date" in df.columns:
        st.subheader("Sentiment Trend Over Time")
        st.plotly_chart(plot_sentiment_over_time(df), use_container_width=True)


# ===========================================================================
# PAGE 3 — LIVE DEMO
# ===========================================================================
elif page == "🤖 Live Demo":
    st.title("🤖 Live Sentiment Prediction")

    model_choice = st.radio(
        "Choose model",
        ["Baseline (TF-IDF + LR)", "BERT Fine-tuned"],
        horizontal=True,
    )

    st.markdown("---")
    input_mode = st.radio("Input mode", ["Single text", "Batch (one per line)"], horizontal=True)

    if input_mode == "Single text":
        user_text = st.text_area("Enter text to analyse", height=120,
                                 placeholder="e.g. I absolutely love this product!")
        texts_to_analyse = [user_text.strip()] if user_text.strip() else []
    else:
        batch_input = st.text_area("Enter texts (one per line)", height=200)
        texts_to_analyse = [t.strip() for t in batch_input.splitlines() if t.strip()]

    analyse_btn = st.button("Analyse ➜", type="primary", disabled=not texts_to_analyse)

    if analyse_btn and texts_to_analyse:
        from src.data_loader import clean_text
        from src.visualize import plot_confidence_gauge

        cleaned = [clean_text(t) for t in texts_to_analyse]

        with st.spinner("Running inference…"):
            try:
                if model_choice.startswith("Baseline"):
                    pipeline = get_baseline_pipeline()
                    if pipeline is None:
                        st.error(
                            "Baseline model not found. "
                            "Train it first with `python src/baseline_model.py`."
                        )
                        st.stop()
                    from src.baseline_model import predict
                    labels, probs = predict(pipeline, cleaned)
                    confidences = probs.max(axis=1)

                else:  # BERT
                    model, tokenizer = get_bert_model()
                    if model is None:
                        st.error(
                            "BERT model not found. "
                            "Train it first with `python src/bert_model.py`."
                        )
                        st.stop()
                    from src.bert_model import predict_bert
                    labels, confidences = predict_bert(model, tokenizer, cleaned)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        label_map = {0: "Negative 😞", 1: "Positive 😊", 2: "Neutral 😐"}
        results = []

        if len(texts_to_analyse) == 1:
            # Single result — rich display
            label = int(labels[0])
            conf = float(confidences[0])
            sentiment_str = label_map.get(label, str(label))

            st.markdown("### Result")
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                color = "green" if label == 1 else "red" if label == 0 else "gray"
                st.markdown(
                    f"<h2 style='color:{color};'>{sentiment_str}</h2>",
                    unsafe_allow_html=True,
                )
                st.progress(conf, text=f"Confidence: {conf:.1%}")
            with res_col2:
                st.plotly_chart(
                    plot_confidence_gauge(conf, sentiment_str.split()[0]),
                    use_container_width=True,
                )
            results.append({
                "text": texts_to_analyse[0],
                "sentiment": sentiment_str,
                "confidence": f"{conf:.2%}",
            })

            # ── SHAP explanation (baseline model only) ──────────────────────
            if model_choice.startswith("Baseline"):
                st.markdown("---")
                st.markdown("### Why this prediction?")
                st.caption(
                    "SHAP token attributions — green bars push toward the "
                    "predicted class, red bars push away."
                )
                try:
                    from src.explain import explain_baseline_prediction, shap_to_plotly_bar
                    pipeline = get_baseline_pipeline()
                    contribs, pred_cls, classes = explain_baseline_prediction(
                        pipeline, cleaned[0]
                    )
                    label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}
                    fig_shap = shap_to_plotly_bar(contribs, pred_cls, label_names)
                    st.plotly_chart(fig_shap, use_container_width=True)
                except ImportError:
                    st.info(
                        "Install `shap` to enable explanations: "
                        "`pip install shap`"
                    )
                except Exception as exc:
                    st.warning(f"Explanation unavailable: {exc}")

        else:
            # Batch — table display
            for orig, label, conf in zip(texts_to_analyse, labels, confidences):
                sentiment_str = label_map.get(int(label), str(label))
                results.append({
                    "text": orig,
                    "sentiment": sentiment_str,
                    "confidence": f"{conf:.2%}",
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Download button
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇ Download results CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )


# ===========================================================================
# PAGE 4 — MODEL COMPARISON
# ===========================================================================
elif page == "📈 Model Comparison":
    st.title("📈 Model Performance Comparison")

    st.markdown(
        """
        The table and chart below compare the **Baseline** (TF-IDF + Logistic Regression)
        and **BERT Fine-tuned** models on the held-out test set.

        > Run the training scripts once to populate real numbers:
        > ```bash
        > python src/baseline_model.py   # trains baseline
        > python src/bert_model.py       # trains BERT
        > ```
        """
    )

    # --- Load saved metrics (populated after training runs) ---
    import json as _json

    _metrics_path = ROOT_DIR / "reports" / "metrics.json"
    _saved: dict = {}
    if _metrics_path.exists():
        try:
            with open(_metrics_path, encoding="utf-8") as _fh:
                _saved = _json.load(_fh)
        except Exception:
            pass

    _empty = {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}
    baseline_results = _saved.get("baseline", _empty)
    bert_results = _saved.get("bert", _empty)

    # Check if saved figures exist and display them
    comparison_img = FIGURES_DIR / "model_comparison.png"
    if comparison_img.exists():
        st.image(str(comparison_img), caption="Model Comparison", use_container_width=True)
    else:
        st.info("No comparison chart found yet — train both models first.")

    st.markdown("---")
    st.subheader("Metrics Summary")

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison_df = pd.DataFrame(
        {
            "Baseline (TF-IDF + LR)": [baseline_results.get(m) for m in metrics],
            "BERT Fine-tuned": [bert_results.get(m) for m in metrics],
        },
        index=[m.upper() for m in metrics],
    )

    if _saved:
        st.dataframe(
            comparison_df.style.format(lambda v: f"{v:.4f}" if v is not None else "—"),
            use_container_width=True,
        )
    else:
        st.info(
            "指标数据尚未生成。请先运行训练脚本：\n"
            "```bash\npython src/baseline_model.py\npython src/bert_model.py\n```"
        )
        st.dataframe(comparison_df.fillna("—"), use_container_width=True)

    st.markdown("---")
    st.subheader("Analysis")
    st.markdown(
        """
        **Baseline (TF-IDF + Logistic Regression)**
        - ✅ Trains in seconds on CPU
        - ✅ Highly interpretable (feature weights are readable)
        - ✅ Low memory footprint
        - ❌ Cannot capture long-range context or word order
        - ❌ Vocabulary mismatch hurts on unseen phrasing

        **BERT Fine-tuned**
        - ✅ State-of-the-art contextual representations
        - ✅ Handles negation, sarcasm, and complex syntax better
        - ✅ Transfer learning — generalises from pre-training on 3.3B words
        - ❌ Requires GPU for practical training speed
        - ❌ Much larger model footprint (110 M parameters)

        **Recommendation:** Use BERT when accuracy is critical and compute is available;
        fall back to the baseline for rapid prototyping or resource-constrained deployments.
        """
    )

    # ROC curves
    roc_baseline = FIGURES_DIR / "roc_curve_baseline_(tf-idf_+_lr).png"
    roc_bert = FIGURES_DIR / "roc_curve_bert_fine-tuned.png"
    col1, col2 = st.columns(2)
    if roc_baseline.exists():
        col1.image(str(roc_baseline), caption="Baseline ROC", use_container_width=True)
    else:
        col1.info("Baseline ROC curve not available yet.")
    if roc_bert.exists():
        col2.image(str(roc_bert), caption="BERT ROC", use_container_width=True)
    else:
        col2.info("BERT ROC curve not available yet.")
