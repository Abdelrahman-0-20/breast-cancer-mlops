"""Streamlit web app for breast-cancer-mlops (milestone M4).

Three tabs: EDA, Model, Predict. All data, training, registry and
evaluation logic is reused from the mlops package -- nothing is
duplicated here beyond thin presentation helpers. Plain design, no
emojis. Run with: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from mlops import config
from mlops.data import load_breast_cancer, split_data
from mlops.evaluate import evaluate_model
from mlops.registry import load_model, load_registry, latest
from mlops.train import run_training

CLASS_NAMES = {0: "benign", 1: "malignant"}


# --- importable helpers (covered by tests/test_app.py) ------------------


def class_distribution(y: np.ndarray) -> pd.DataFrame:
    """Counts per class as a small labelled frame."""
    counts = pd.Series(np.asarray(y)).value_counts().sort_index()
    return pd.DataFrame(
        {"class": [CLASS_NAMES[int(i)] for i in counts.index], "count": counts.to_numpy()}
    )


def feature_frame(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Features plus a human-readable class column."""
    frame = pd.DataFrame(np.asarray(X), columns=list(feature_names))
    frame["class"] = [CLASS_NAMES[int(v)] for v in np.asarray(y)]
    return frame


def correlation_table(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Rounded feature correlation matrix."""
    return frame[list(feature_names)].corr().round(2)


def feature_distribution(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Counts per class for binned feature values.

    Bin labels are plain strings on purpose: an IntervalIndex cannot be
    serialized by Altair and makes st.bar_chart raise a schema error.
    """
    binned = pd.cut(frame[feature], bins=15).astype(str)
    return frame.groupby([binned, "class"], observed=True).size().unstack(fill_value=0)


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """2x2 confusion matrix with readable labels."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=[f"actual {CLASS_NAMES[0]}", f"actual {CLASS_NAMES[1]}"],
        columns=[f"predicted {CLASS_NAMES[0]}", f"predicted {CLASS_NAMES[1]}"],
    )


def verdict_text(label: int, probability: float) -> str:
    """One plain sentence for the prediction result (ascii only, no emojis)."""
    if label == 1:
        return (
            f"The model classifies this case as malignant with {probability:.0%} "
            "probability. Specialist review is recommended."
        )
    return (
        f"The model classifies this case as benign with {probability:.0%} "
        "probability. Routine follow-up applies."
    )


def predict_one(model: object, row: np.ndarray) -> tuple[int, float]:
    """Predict one sample: (label, probability of malignancy)."""
    array = np.asarray(row, dtype=float).reshape(1, -1)
    label = int(model.predict(array)[0])
    probability = float(model.predict_proba(array)[0, 1])
    return label, probability


@st.cache_data(show_spinner=False)
def dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The canonical dataset, cached across reruns."""
    return load_breast_cancer()


@st.cache_resource(show_spinner=False)
def model_bundle() -> tuple[object, dict, str]:
    """Latest registered model, or a canonical seeded training run."""
    entries = load_registry()
    if entries:
        entry = latest()
        model = load_model(int(entry["version"]))
        metrics = dict(entry.get("metrics", {}))
        return model, metrics, f"registry: model-v{entry['version']}"
    run = run_training(seed=config.SEED)
    source = f"trained in memory (seed {config.SEED}); no registry entry yet"
    return run["model"], run["metrics"], source


def evaluate_current(model: object) -> dict:
    """Holdout evaluation of the current model against the gate."""
    X, y, _ = dataset()
    _, _, X_test, _, _, y_test = split_data(X, y, seed=config.SEED)
    return evaluate_model(model, X_test, y_test)


def holdout_split() -> tuple[np.ndarray, np.ndarray]:
    """The seeded holdout test split."""
    X, y, _ = dataset()
    _, _, X_test, _, _, y_test = split_data(X, y, seed=config.SEED)
    return X_test, y_test


# --- UI (only when executed by streamlit, not when imported by tests) ---


def main() -> None:
    st.set_page_config(page_title="Breast Cancer MLOps", layout="wide")
    st.title("Breast Cancer MLOps")
    st.caption("EDA, model metrics, and prediction, built on this repository's mlops package.")

    X, y, names = dataset()
    model, _, source = model_bundle()

    tab_eda, tab_model, tab_predict = st.tabs(["EDA", "Model", "Predict"])

    with tab_eda:
        st.subheader("Class distribution")
        st.bar_chart(class_distribution(y).set_index("class"))

        st.subheader("Feature distribution by class")
        feature = st.selectbox("Feature", names)
        frame = feature_frame(X, y, names)
        st.bar_chart(feature_distribution(frame, feature))

        st.subheader("Correlation matrix")
        st.dataframe(correlation_table(frame, names), use_container_width=True)

    with tab_model:
        st.subheader("Model in use")
        st.write(source)
        result = evaluate_current(model)
        metrics = result["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Holdout accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.4f}")
        col3.metric("Evaluation gate", "passed" if result["passed"] else "failed")
        st.caption(
            f"Thresholds from mlops/config.py: accuracy >= {config.MIN_TEST_ACCURACY}, "
            f"ROC-AUC >= {config.MIN_TEST_ROC_AUC}"
        )
        if result["passed"]:
            st.success("All thresholds met on the holdout test split.")
        else:
            st.error("\n".join(result["failures"]))

        st.subheader("Confusion matrix (holdout test)")
        X_test, y_test = holdout_split()
        y_pred = model.predict(X_test)
        st.dataframe(confusion_table(y_test, y_pred), use_container_width=True)

    with tab_predict:
        st.subheader("Predict from feature values")
        st.caption("Defaults are the dataset means; adjust any value and press Predict.")
        means = X.mean(axis=0)
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        columns = st.columns(3)
        inputs: list[float] = []
        for index, name in enumerate(names):
            with columns[index % 3]:
                value = st.number_input(
                    name,
                    min_value=float(mins[index]),
                    max_value=float(maxs[index]),
                    value=float(means[index]),
                    key=f"feature_{index}",
                )
            inputs.append(float(value))
        if st.button("Predict"):
            label, probability = predict_one(model, inputs)
            st.metric("Probability of malignancy", f"{probability:.1%}")
            st.write(verdict_text(label, probability))


if __name__ == "__main__":
    main()
