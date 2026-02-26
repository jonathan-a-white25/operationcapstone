"""
app.py  ─  Sentiment Analysis Dashboard
========================================
Two modes:
  1. Predict  – enter text → get sentiment + confidence
  2. EDA      – upload CSV/JSONL → exploratory analysis + batch prediction
"""

import os
import io
import re
import html as html_lib
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="SentimentOS",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Import model utilities ─────────────────────────────────────────────────
from model.sentiment_model import (
    train_and_save,
    load_model,
    predict,
    clean_text,
)

# ── Plotly dark template ───────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.8)",
    font_family="Space Mono",
    font_color="#E2E8F0",
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#00F5C4", "#FF4D6D", "#FFD166", "#7C6AF7", "#00B4D8"],
)

# ── Helper: load CSS-injected HTML snippets ────────────────────────────────
def pill(label: str, variant: str = "neutral") -> str:
    icons = {"positive": "▲", "negative": "▼", "neutral": "●"}
    return (
        f'<span class="metric-pill pill-{variant}">'
        f'{icons.get(variant, "●")} {label}</span>'
    )

def conf_bar(confidence: float, label: str) -> str:
    pct = int(confidence * 100)
    cls = "conf-bar-positive" if label == "positive" else "conf-bar-negative"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill {cls}" style="width:{pct}%"></div>'
        f'</div>'
    )

def result_card(item: dict) -> str:
    label  = item["label"]
    conf   = item["confidence"]
    text   = item["text"][:160] + ("…" if len(item["text"]) > 160 else "")
    conf_p = f"{conf*100:.1f}%"
    return (
        f'<div class="card card-{label}">'
        f'  <div style="margin-bottom:0.4rem">{pill(label.upper(), label)} '
        f'    <span style="font-size:0.72rem;color:#64748b;font-family:Space Mono,monospace">'
        f'    confidence: {conf_p}</span></div>'
        f'  <div style="font-size:0.82rem;color:#E2E8F0;font-family:Space Mono,monospace;'
        f'    line-height:1.55">{text}</div>'
        f'  {conf_bar(conf, label)}'
        f'</div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="app-title" style="font-size:1.4rem">🔮 Sentiment<span>OS</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">v1.0 · Amazon Fashion NLP</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    mode = st.radio(
        "MODE",
        ["📝  Predict", "📊  EDA & Batch"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)

    pipeline, threshold = load_model()
    model_loaded = pipeline is not None

    if model_loaded:
        st.success(f"✓ Model loaded  (t={threshold:.2f})")
    else:
        st.warning("No trained model found.\nTrain one in the EDA tab.")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#334155;font-family:Space Mono,monospace">'
        'TF-IDF · LogisticRegression<br>ngram (1,2) · max_features=100k<br>'
        'threshold sweep · macro-F1</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '  <div class="app-title">Sentiment<span>OS</span></div>'
    '  <div class="app-subtitle">// NLP · Amazon Fashion · TF-IDF + Logistic Regression</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# MODE 1 – PREDICT
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📝  Predict":

    st.markdown('<div class="section-header">Single or Bulk Text Prediction</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️  No model loaded. Switch to **EDA & Batch** to train one first.")
        st.stop()

    # ── Input area ────────────────────────────────────────────────────────
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("**Enter one review per line:**")
        user_text = st.text_area(
            label="reviews",
            label_visibility="collapsed",
            placeholder="e.g.\nLove this jacket, fits perfectly!\nTerrible quality, fell apart after one wash.",
            height=220,
        )
        run_btn = st.button("⚡  ANALYZE", use_container_width=True)

    with col_result:
        if run_btn and user_text.strip():
            lines = [l.strip() for l in user_text.strip().splitlines() if l.strip()]
            with st.spinner("Running inference…"):
                results = predict(lines, pipeline, threshold)

            pos = sum(1 for r in results if r["label"] == "positive")
            neg = len(results) - pos

            # Quick stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(results))
            c2.metric("Positive", pos)
            c3.metric("Negative", neg)

            st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)
            for item in results:
                st.markdown(result_card(item), unsafe_allow_html=True)

        elif run_btn:
            st.info("Please enter at least one review.")
        else:
            st.markdown(
                '<div style="color:#334155;font-family:Space Mono,monospace;'
                'font-size:0.8rem;padding-top:3rem;text-align:center">'
                '← type a review and hit ANALYZE<span class="blink">_</span></div>',
                unsafe_allow_html=True,
            )

    # ── Demo examples ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Quick Demo Examples</div>', unsafe_allow_html=True)
    demo_cols = st.columns(3)
    demos = [
        ("Positive 🟢", "Absolutely love this bag! Great quality stitching and the color is stunning. Would buy again."),
        ("Negative 🔴", "Complete waste of money. The zipper broke after a week and the material feels cheap."),
        ("Mixed 🟡", "It looks okay in photos but the actual item is smaller than expected. Shipping was fast though."),
    ]
    for col, (label, text) in zip(demo_cols, demos):
        with col:
            if st.button(label, use_container_width=True, key=f"demo_{label}"):
                res = predict([text], pipeline, threshold)[0]
                st.markdown(result_card(res), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODE 2 – EDA & BATCH
# ═════════════════════════════════════════════════════════════════════════════
else:
    tab_eda, tab_train, tab_batch = st.tabs(
        ["📈  Exploratory Analysis", "🏋️  Train Model", "🗂️  Batch Predict"]
    )

    # ── Helper: load uploaded data ─────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_uploaded(reviews_bytes, meta_bytes, nrows):
        reviews = pd.read_json(io.BytesIO(reviews_bytes), lines=True, nrows=nrows)
        metadata = pd.read_json(io.BytesIO(meta_bytes), lines=True, nrows=nrows)
        df = reviews.merge(metadata, how="left", on="parent_asin", suffixes=("_review", "_meta"))
        cols = [c for c in ["title_meta", "rating", "parent_asin", "title_review", "text"] if c in df.columns]
        df = df[cols]
        df["clean_text"] = df["text"].apply(clean_text)
        return df

    # ── EDA TAB ────────────────────────────────────────────────────────────
    with tab_eda:
        st.markdown('<div class="section-header">Upload your JSONL files</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            reviews_file = st.file_uploader("Reviews JSONL", type=["jsonl", "json"], key="rev")
        with c2:
            meta_file = st.file_uploader("Metadata JSONL", type=["jsonl", "json"], key="meta")

        nrows = st.slider("Max rows to load", 10_000, 500_000, 100_000, step=10_000)

        if reviews_file and meta_file:
            with st.spinner("Loading & merging datasets…"):
                df = load_uploaded(reviews_file.read(), meta_file.read(), nrows)
            st.session_state["df"] = df
            st.success(f"✓ Loaded {len(df):,} rows · {df.shape[1]} columns")

            # ── TextBlob sentiment (sample for speed) ─────────────────────
            if "sentiment" not in df.columns:
                sample_n = min(20_000, len(df))
                sample_df = df.sample(sample_n, random_state=42).copy()
                try:
                    from textblob import TextBlob
                    with st.spinner("Computing TextBlob polarity (sample)…"):
                        sample_df["polarity"] = sample_df["text"].apply(
                            lambda x: TextBlob(str(x)).sentiment.polarity
                        )
                        sample_df["sentiment"] = sample_df["polarity"].apply(
                            lambda p: "positive" if p > 0 else ("negative" if p < 0 else "neutral")
                        )
                    df = df.merge(sample_df[["polarity", "sentiment"]], left_index=True, right_index=True, how="left")
                except ImportError:
                    df["polarity"] = np.nan
                    df["sentiment"] = "unknown"

            st.session_state["df"] = df
            st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

            # ── KPI row ───────────────────────────────────────────────────
            st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Reviews", f"{len(df):,}")
            k2.metric("Unique Products", f"{df['parent_asin'].nunique():,}" if "parent_asin" in df.columns else "—")
            k3.metric("Avg Rating", f"{df['rating'].mean():.2f}" if "rating" in df.columns else "—")
            k4.metric("Avg Tokens", f"{df['clean_text'].apply(lambda x: len(str(x).split())).mean():.0f}")

            st.markdown("---")

            # ── Charts ────────────────────────────────────────────────────
            plot_c1, plot_c2 = st.columns(2)

            with plot_c1:
                if "rating" in df.columns:
                    rating_counts = df["rating"].value_counts().sort_index()
                    fig = go.Figure(go.Bar(
                        x=rating_counts.index.astype(str),
                        y=rating_counts.values,
                        marker_color=["#FF4D6D", "#FF8C42", "#FFD166", "#7BC67E", "#00F5C4"],
                        text=rating_counts.values,
                        textposition="outside",
                    ))
                    fig.update_layout(title="Rating Distribution", xaxis_title="Stars",
                                      yaxis_title="Count", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            with plot_c2:
                if "sentiment" in df.columns and df["sentiment"].notna().any():
                    sent_counts = df["sentiment"].value_counts()
                    color_map = {"positive": "#00F5C4", "negative": "#FF4D6D", "neutral": "#FFD166", "unknown": "#64748B"}
                    colors = [color_map.get(s, "#64748B") for s in sent_counts.index]
                    fig = go.Figure(go.Pie(
                        labels=sent_counts.index,
                        values=sent_counts.values,
                        hole=0.55,
                        marker_colors=colors,
                    ))
                    fig.update_layout(title="Sentiment Distribution (TextBlob sample)", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            # ── Avg polarity by rating ─────────────────────────────────────
            if "polarity" in df.columns and df["polarity"].notna().any():
                avg_pol = df.groupby("rating")["polarity"].mean().reset_index()
                fig = px.bar(
                    avg_pol, x="rating", y="polarity",
                    color="polarity",
                    color_continuous_scale=["#FF4D6D", "#FFD166", "#00F5C4"],
                    title="Avg TextBlob Polarity by Rating",
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # ── Token distribution ─────────────────────────────────────────
            df["token_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
            fig = px.histogram(
                df[df["token_count"] < 300], x="token_count", nbins=60,
                title="Token Count Distribution",
                color_discrete_sequence=["#00F5C4"],
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            # ── Top products ───────────────────────────────────────────────
            if "parent_asin" in df.columns:
                top = (
                    df.groupby("parent_asin")
                    .agg(reviews=("rating", "count"), avg_rating=("rating", "mean"),
                         title=("title_meta", "first") if "title_meta" in df.columns else ("parent_asin", "first"))
                    .sort_values("reviews", ascending=False)
                    .head(10)
                    .reset_index()
                )
                fig = px.bar(
                    top, x="reviews", y="parent_asin",
                    orientation="h",
                    color="avg_rating",
                    color_continuous_scale=["#FF4D6D", "#FFD166", "#00F5C4"],
                    title="Top 10 Products by Review Count",
                    hover_data=["title"] if "title" in top.columns else [],
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # ── Raw data preview ───────────────────────────────────────────
            with st.expander("🔍 Raw Data Preview"):
                st.dataframe(df.head(100), use_container_width=True)

        else:
            st.info("Upload both `Amazon_Fashion.jsonl` and `meta_Amazon_Fashion.jsonl` to begin.")

    # ── TRAIN TAB ──────────────────────────────────────────────────────────
    with tab_train:
        st.markdown('<div class="section-header">Train the TF-IDF + LR Pipeline</div>', unsafe_allow_html=True)

        if "df" not in st.session_state:
            st.warning("Load your data in the **Exploratory Analysis** tab first.")
        else:
            df = st.session_state["df"]
            st.info(f"Using {len(df):,} rows from loaded dataset.\n\nRatings 1-2 → **negative**, 4-5 → **positive** (rating 3 excluded).")

            if st.button("🚀  START TRAINING", use_container_width=True):
                progress = st.progress(0, text="Preparing data…")
                time.sleep(0.3)
                progress.progress(15, text="Splitting train/val/test…")

                try:
                    with st.spinner("Training in progress (this may take a minute)…"):
                        pipeline_, threshold_, report, cm = train_and_save(df)

                    progress.progress(100, text="Done!")
                    st.success(f"✓ Model trained! Optimal threshold: **{threshold_:.2f}**")

                    # Update sidebar state
                    pipeline, threshold = pipeline_, threshold_
                    model_loaded = True

                    # Confusion matrix
                    cm_fig = go.Figure(go.Heatmap(
                        z=cm,
                        x=["Pred: Neg", "Pred: Pos"],
                        y=["True: Neg", "True: Pos"],
                        colorscale=[[0, "#0A0E1A"], [1, "#00F5C4"]],
                        text=cm,
                        texttemplate="%{text}",
                        showscale=False,
                    ))
                    cm_fig.update_layout(title="Confusion Matrix (test set)", **PLOTLY_LAYOUT)
                    st.plotly_chart(cm_fig, use_container_width=True)

                    with st.expander("📋 Full Classification Report"):
                        st.code(report, language=None)

                except Exception as e:
                    st.error(f"Training failed: {e}")
                    progress.empty()

    # ── BATCH PREDICT TAB ──────────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="section-header">Batch Predict from CSV/JSONL</div>', unsafe_allow_html=True)

        if not model_loaded:
            st.error("Train a model first (EDA tab → Train tab).")
        else:
            batch_file = st.file_uploader("Upload CSV or JSONL with a text column", type=["csv", "jsonl", "json"])

            if batch_file:
                fname = batch_file.name
                if fname.endswith(".csv"):
                    batch_df = pd.read_csv(batch_file)
                else:
                    batch_df = pd.read_json(batch_file, lines=True)

                st.write(f"**Shape:** {batch_df.shape}")
                st.dataframe(batch_df.head(5), use_container_width=True)

                text_col = st.selectbox("Which column contains the review text?", batch_df.columns.tolist())

                if st.button("⚡  RUN BATCH PREDICTION", use_container_width=True):
                    with st.spinner(f"Predicting {len(batch_df):,} rows…"):
                        texts = batch_df[text_col].fillna("").astype(str).tolist()
                        results = predict(texts, pipeline, threshold)

                    batch_df["sentiment_label"] = [r["label"] for r in results]
                    batch_df["confidence"]       = [r["confidence"] for r in results]
                    batch_df["raw_prob_pos"]     = [r["raw_prob"] for r in results]

                    # Summary
                    counts = batch_df["sentiment_label"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total", len(batch_df))
                    c2.metric("Positive", counts.get("positive", 0))
                    c3.metric("Negative", counts.get("negative", 0))

                    # Chart
                    fig = px.histogram(
                        batch_df, x="raw_prob_pos", nbins=50,
                        title="Distribution of Positive Probability",
                        color_discrete_sequence=["#00F5C4"],
                    )
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#FF4D6D",
                                  annotation_text=f"threshold={threshold:.2f}")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                    # Download
                    csv_out = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️  Download Results CSV",
                        csv_out,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    with st.expander("🔍 Preview Results"):
                        st.dataframe(batch_df.head(50), use_container_width=True)
