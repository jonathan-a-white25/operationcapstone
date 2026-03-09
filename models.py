"""
models.py
---------
Three sentiment models:
  1. Baseline  — TF-IDF + Logistic Regression (your original)
  2. Enhanced  — TF-IDF (word + char n-grams) + LinearSVC  (Claude's model)
  3. DistilBERT — pretrained HuggingFace transformer, zero training needed

All three expose the same interface:
    train(X_train, y_train) -> fitted model
    predict_proba(X) -> np.array of positive-class probabilities  (or scores)
    predict(X, threshold) -> np.array of 0/1 labels
"""

import re
import html
import numpy as np
import pickle
import os

# ── Text cleaning ──────────────────────────────────────────────────────────
CONTRACTION_MAP = {
    "don't": "do not", "can't": "cannot", "won't": "will not",
    "it's": "it is", "i'm": "i am", "you're": "you are",
    "they're": "they are", "we're": "we are", "that's": "that is",
}

def expand_contractions(text):
    for c, full in CONTRACTION_MAP.items():
        text = text.replace(c, full)
    return text

def clean_text(text):
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

# ── Data prep (shared) ─────────────────────────────────────────────────────
def prepare_data(df):
    """Binary split: rating>=4 -> 1, rating<=2 -> 0, drop 3."""
    from sklearn.model_selection import train_test_split
    d = df[df["rating"].isin([1, 2, 4, 5])].copy()
    d["y"] = (d["rating"] >= 4).astype(int)
    X = d["clean_text"].fillna("").astype(str).to_numpy(dtype=object)
    y = d["y"].to_numpy(dtype=int)
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)
    return X_train, X_val, X_test, y_train, y_val, y_test

def best_threshold(proba, y_true):
    from sklearn.metrics import f1_score
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (proba >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

def evaluate(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, confusion_matrix, classification_report
    )
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "f1_macro":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_pos":    round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_neg":    round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report":    classification_report(y_true, y_pred, digits=4, zero_division=0),
    }

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — TF-IDF + Logistic Regression (baseline)
# ══════════════════════════════════════════════════════════════════════════════
def train_logistic(df):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2)),
        ("clf",   LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga")),
    ])
    pipe.fit(X_train, y_train)

    t = best_threshold(pipe.predict_proba(X_val)[:, 1], y_val)
    y_pred = (pipe.predict_proba(X_test)[:, 1] >= t).astype(int)
    metrics = evaluate(y_test, y_pred)
    metrics["threshold"] = t
    metrics["y_test"] = y_test.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["proba"]  = pipe.predict_proba(X_test)[:, 1].tolist()

    return pipe, t, metrics


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — TF-IDF (word + char n-grams) + LinearSVC  [Claude's model]
# ══════════════════════════════════════════════════════════════════════════════
def train_svc(df):
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    # Combined word-level AND character-level TF-IDF features
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            max_features=50_000,
            ngram_range=(3, 5),
            min_df=3,
            sublinear_tf=True,
        )),
    ])

    # CalibratedClassifierCV wraps LinearSVC to give us probabilities
    pipe = Pipeline([
        ("features", features),
        ("clf", CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight="balanced", C=0.5),
            cv=3,
        )),
    ])
    pipe.fit(X_train, y_train)

    t = best_threshold(pipe.predict_proba(X_val)[:, 1], y_val)
    y_pred = (pipe.predict_proba(X_test)[:, 1] >= t).astype(int)
    metrics = evaluate(y_test, y_pred)
    metrics["threshold"] = t
    metrics["y_test"] = y_test.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["proba"]  = pipe.predict_proba(X_test)[:, 1].tolist()

    return pipe, t, metrics


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — DistilBERT (pretrained, no training needed)
# ══════════════════════════════════════════════════════════════════════════════
def load_distilbert():
    """Load pretrained DistilBERT sentiment pipeline from HuggingFace."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )

def evaluate_distilbert(df, hf_pipe, batch_size=64):
    """Run DistilBERT on the test split and return metrics."""
    d = df[df["rating"].isin([1, 2, 4, 5])].copy()
    d["y"] = (d["rating"] >= 4).astype(int)

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        d["text"].fillna("").astype(str).to_numpy(dtype=object),
        d["y"].to_numpy(dtype=int),
        test_size=0.2, random_state=42, stratify=d["y"].to_numpy(dtype=int)
    )

    # Cap at 1,000 rows for speed on free tier CPU
    if len(X_test) > 1000:
        idx = np.random.default_rng(42).choice(len(X_test), 1000, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]


    # Batch inference
    results, probas = [], []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size].tolist()
        out   = hf_pipe(batch)
        for r in out:
            label = 1 if r["label"] == "POSITIVE" else 0
            prob  = r["score"] if r["label"] == "POSITIVE" else 1 - r["score"]
            results.append(label)
            probas.append(prob)

    y_pred = np.array(results)
    metrics = evaluate(y_test, y_pred)
    metrics["threshold"] = 0.5
    metrics["y_test"] = y_test.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["proba"]  = probas

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Predict (single or batch) — used by the Predict page
# ══════════════════════════════════════════════════════════════════════════════
def predict_all(texts, lr_pipe, lr_t, svc_pipe, svc_t, hf_pipe):
    """Run all 3 models on a list of texts. Returns list of dicts."""
    if isinstance(texts, str):
        texts = [texts]

    cleaned = [clean_text(t) for t in texts]

    lr_probas  = lr_pipe.predict_proba(cleaned)[:, 1]
    svc_probas = svc_pipe.predict_proba(cleaned)[:, 1]
    hf_out     = hf_pipe(texts, truncation=True, max_length=512)

    results = []
    for i, text in enumerate(texts):
        lr_prob  = float(lr_probas[i])
        svc_prob = float(svc_probas[i])
        hf_prob  = hf_out[i]["score"] if hf_out[i]["label"] == "POSITIVE" else 1 - hf_out[i]["score"]

        results.append({
            "text": text,
            "lr":  {"label": "positive" if lr_prob  >= lr_t  else "negative", "confidence": lr_prob  if lr_prob  >= lr_t  else 1 - lr_prob,  "raw_prob": lr_prob},
            "svc": {"label": "positive" if svc_prob >= svc_t else "negative", "confidence": svc_prob if svc_prob >= svc_t else 1 - svc_prob, "raw_prob": svc_prob},
            "bert":{"label": "positive" if hf_out[i]["label"] == "POSITIVE" else "negative", "confidence": float(hf_out[i]["score"]), "raw_prob": float(hf_prob)},
        })
    return results
