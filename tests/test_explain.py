"""Tests for src/explain.py — SHAP baseline explanations."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

shap = pytest.importorskip("shap", reason="shap not installed")


@pytest.fixture(scope="module")
def trained_pipeline():
    from src.data_loader import generate_mock_data, preprocess_dataframe, split_data
    from src.baseline_model import train_baseline
    import tempfile
    import pathlib
    df = preprocess_dataframe(generate_mock_data(n=150))
    tmp = pathlib.Path(tempfile.mkdtemp())
    train, val, _ = split_data(df, save_dir=tmp)
    return train_baseline(train, val)


class TestExplainBaselinePrediction:
    def test_returns_tuple_of_three(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        result = explain_baseline_prediction(trained_pipeline, "great product love it")
        assert len(result) == 3

    def test_contributions_is_list(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        contribs, _, _ = explain_baseline_prediction(trained_pipeline, "love this")
        assert isinstance(contribs, list)

    def test_each_contribution_is_tuple(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        contribs, _, _ = explain_baseline_prediction(trained_pipeline, "love this product")
        for token, value in contribs:
            assert isinstance(token, str)
            assert isinstance(value, float)

    def test_predicted_class_is_int(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        _, pred, _ = explain_baseline_prediction(trained_pipeline, "terrible service")
        assert isinstance(pred, int)

    def test_predicted_class_in_known_classes(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        _, pred, classes = explain_baseline_prediction(trained_pipeline, "okay day")
        assert pred in classes

    def test_n_top_respected(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        contribs, _, _ = explain_baseline_prediction(
            trained_pipeline, "absolutely love this amazing product!", n_top=5
        )
        assert len(contribs) <= 5

    def test_sorted_by_abs_value(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        contribs, _, _ = explain_baseline_prediction(
            trained_pipeline, "love this wonderful amazing product"
        )
        values = [abs(v) for _, v in contribs]
        assert values == sorted(values, reverse=True)

    def test_empty_text_does_not_crash(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        contribs, pred, classes = explain_baseline_prediction(trained_pipeline, "")
        assert isinstance(contribs, list)

    def test_stopword_only_text(self, trained_pipeline):
        from src.explain import explain_baseline_prediction
        contribs, _, _ = explain_baseline_prediction(trained_pipeline, "the a is")
        assert isinstance(contribs, list)


class TestShapToPlotlyBar:
    def test_returns_figure(self):
        import plotly.graph_objects as go
        from src.explain import shap_to_plotly_bar
        contribs = [("love", 0.42), ("product", 0.31), ("bad", -0.15)]
        fig = shap_to_plotly_bar(contribs, predicted_class=1)
        assert isinstance(fig, go.Figure)

    def test_empty_contributions_returns_figure(self):
        import plotly.graph_objects as go
        from src.explain import shap_to_plotly_bar
        fig = shap_to_plotly_bar([], predicted_class=0)
        assert isinstance(fig, go.Figure)

    def test_title_contains_class_name(self):
        from src.explain import shap_to_plotly_bar
        fig = shap_to_plotly_bar([("great", 0.5)], predicted_class=1)
        assert "Positive" in fig.layout.title.text

    def test_custom_label_names(self):
        from src.explain import shap_to_plotly_bar
        fig = shap_to_plotly_bar(
            [("good", 0.3)],
            predicted_class=0,
            label_names={0: "Happy", 1: "Sad"},
        )
        assert "Happy" in fig.layout.title.text
