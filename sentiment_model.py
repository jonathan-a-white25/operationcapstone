"""
sentiment_model.py
------------------
Train the TF-IDF + Logistic Regression sentiment model and expose
a single predict() function for the Streamlit app.
"""

import re
import html
import pickle
import os
import numpy as np
from collections import Counter

# ── Contraction map ────────────────────────────────────────────────────────
CONTRACTION_MAP = {
    "don't": "do not", "can't": "cannot", "won't": "will not",
    "it's": "it is", "i'm": "i am", "you're": "you are",
    "they're": "they are", "we're": "we are", "that's": "that is",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")


# ── Text cleaning (mirrors your notebook) ──────────────────────────────────
def expand_contractions(text: str) -> str:
    for contraction, full in CONTRACTION_MAP.items():
        text = text.replace(contraction, full)
    return text


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Train & save ───────────────────────────────────────────────────────────
def train_and_save(df):
    """
    Receives the merged DataFrame (must have 'clean_text' and 'rating').
    Trains the baseline pipeline and pickles it to MODEL_PATH.
    Returns the trained pipeline.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    df_model = df[df["rating"].isin([1, 2, 4, 5])].copy()
    df_model["y"] = (df_model["rating"] >= 4).astype(int)

    X = df_model["clean_text"].fillna("").astype(str).to_numpy(dtype=object)
    y = df_model["y"].to_numpy(dtype=int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Threshold sweep on validation
    proba_val = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_val, (proba_val >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    # Evaluate on test
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= best_t).astype(int)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Persist
    payload = {"pipeline": pipeline, "threshold": best_t}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    return pipeline, best_t, report, cm


# ── Load ───────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["pipeline"], payload["threshold"]


# ── Predict (single string or list) ───────────────────────────────────────
def predict(texts, pipeline=None, threshold=None):
    """
    texts       : str | list[str]
    Returns     : list of dicts with keys: text, label, confidence, polarity
    """
    if pipeline is None:
        pipeline, threshold = load_model()
        if pipeline is None:
            raise RuntimeError("No trained model found. Train a model first.")

    if isinstance(texts, str):
        texts = [texts]

    cleaned = [clean_text(t) for t in texts]
    probas = pipeline.predict_proba(cleaned)[:, 1]
    results = []
    for original, prob in zip(texts, probas):
        label = "positive" if prob >= threshold else "negative"
        results.append({
            "text": original,
            "label": label,
            "confidence": float(prob) if label == "positive" else float(1 - prob),
            "raw_prob": float(prob),
        })
    return results
