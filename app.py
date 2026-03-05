"""
app.py — Sentiment & Review Analysis Dashboard
================================================
Pages:
  1. 📊  Analysis      — EDA dashboard matching mockup
  2. 🤖  Model Compare — run all 3 models, compare metrics
  3. 📝  Predict       — live inference with all 3 models
"""

import os, sys, io, time, importlib.util
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SentimentOS",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

css_path = os.path.join(BASE_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_mfile = os.path.join(BASE_DIR, "models.py")
if not os.path.exists(_mfile):
    st.error(f"models.py not found at {_mfile}")
    st.stop()
_spec = importlib.util.spec_from_file_location("models", _mfile)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["models"] = _mod
_spec.loader.exec_module(_mod)

clean_text          = _mod.clean_text
train_logistic      = _mod.train_logistic
train_svc           = _mod.train_svc
load_distilbert     = _mod.load_distilbert
evaluate_distilbert = _mod.evaluate_distilbert
predict_all         = _mod.predict_all

PL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.8)",
    font_family="Space Mono",
    font_color="#E2E8F0",
    margin=dict(l=20, r=20, t=45, b=20),
)
ACCENT = "#00F5C4"
RED    = "#FF4D6D"
YELLOW = "#FFD166"
BLUE   = "#00B4D8"
PURPLE = "#7C6AF7"

def pill(label, variant="neutral"):
    icons = {"positive":"▲","negative":"▼","neutral":"●"}
    return f'<span class="metric-pill pill-{variant}">{icons.get(variant,"●")} {label}</span>'

def conf_bar(conf, label):
    pct = int(conf * 100)
    cls = "conf-bar-positive" if label == "positive" else "conf-bar-negative"
    return f'<div class="conf-bar-wrap"><div class="conf-bar-fill {cls}" style="width:{pct}%"></div></div>'

def result_card(item, model_key, model_name):
    r     = item[model_key]
    label = r["label"]
    conf  = r["confidence"]
    return (
        f'<div class="card card-{label}" style="margin-bottom:0.5rem">'
        f'<div style="font-size:0.65rem;color:#64748b;font-family:Space Mono,monospace;margin-bottom:0.3rem">{model_name}</div>'
        f'<div style="margin-bottom:0.3rem">{pill(label.upper(), label)} '
        f'<span style="font-size:0.7rem;color:#64748b;font-family:Space Mono,monospace">{conf*100:.1f}%</span></div>'
        f'{conf_bar(conf, label)}</div>'
    )

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].apply(clean_text)
    return df

@st.cache_data(show_spinner=False)
def load_jsonl(rev_bytes, meta_bytes, nrows):
    rev  = pd.read_json(io.BytesIO(rev_bytes),  lines=True, nrows=nrows)
    meta = pd.read_json(io.BytesIO(meta_bytes), lines=True, nrows=nrows)
    df   = rev.merge(meta, how="left", on="parent_asin", suffixes=("_review","_meta"))
    cols = [c for c in ["title_meta","rating","parent_asin","title_review","text"] if c in df.columns]
    df   = df[cols]
    df["clean_text"] = df["text"].apply(clean_text)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title" style="font-size:1.3rem">🔮 Sentiment<span>OS</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">v2.0 · Amazon Fashion NLP</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("NAVIGATE", ["📊  Analysis", "🤖  Model Compare", "📝  Predict"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)

    upload_type = st.radio("File type", ["CSV", "JSONL"], horizontal=True)
    df = st.session_state.get("df", None)

    if upload_type == "CSV":
        up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if up:
            with st.spinner("Loading…"):
                df = load_csv(up.read())
            st.session_state["df"] = df
    else:
        r_up = st.file_uploader("Reviews JSONL",  type=["jsonl","json"], key="rev")
        m_up = st.file_uploader("Metadata JSONL", type=["jsonl","json"], key="meta")
        nrows = st.slider("Max rows", 10_000, 300_000, 100_000, step=10_000)
        if r_up and m_up:
            with st.spinner("Loading…"):
                df = load_jsonl(r_up.read(), m_up.read(), nrows)
            st.session_state["df"] = df

    if df is not None:
        st.success(f"✓ {len(df):,} rows loaded")
    else:
        st.info("Upload a dataset to begin.")

    st.markdown("---")
    st.markdown('<div class="section-header">LR Threshold</div>', unsafe_allow_html=True)
    lr_threshold = st.slider("", 0.10, 0.90,
                             float(st.session_state.get("lr_t", 0.5)),
                             step=0.05, key="lr_threshold_slider")
    st.caption(f"Current: {lr_threshold:.2f}")
    st.markdown("---")
    st.markdown('<div style="font-size:0.62rem;color:#1e293b;font-family:Space Mono,monospace">Model 1: TF-IDF + LR<br>Model 2: TF-IDF + LinearSVC<br>Model 3: DistilBERT</div>', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Sentiment<span>OS</span></div>'
    '<div class="app-subtitle">// Sentiment & Review Analysis · Amazon Fashion · 3-Model Comparison</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Analysis":
    if df is None:
        st.info("⬅️  Upload a dataset in the sidebar to begin.")
        st.stop()

    if "polarity" not in df.columns:
        try:
            from textblob import TextBlob
            sample = df.sample(min(20_000, len(df)), random_state=42).copy()
            sample["polarity"]  = sample["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            sample["sentiment"] = sample["polarity"].apply(lambda p: "positive" if p > 0 else ("negative" if p < 0 else "neutral"))
            df = df.merge(sample[["polarity","sentiment"]], left_index=True, right_index=True, how="left")
            st.session_state["df"] = df
        except Exception:
            df["polarity"]  = 0.0
            df["sentiment"] = "unknown"

    if "sentiment" not in df.columns:
        df["sentiment"] = "unknown"

    df["doc_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))

    total   = len(df)
    pos_pct = (df["sentiment"] == "positive").sum() / total * 100 if total else 0
    neg_pct = (df["sentiment"] == "negative").sum() / total * 100 if total else 0

    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Reviews",   f"{total:,}")
    k2.metric("Avg Rating",      f"{df['rating'].mean():.2f}" if "rating" in df.columns else "—")
    k3.metric("Positive",        f"{pos_pct:.1f}%")
    k4.metric("Negative",        f"{neg_pct:.1f}%")
    k5.metric("Unique Products", f"{df['parent_asin'].nunique():,}" if "parent_asin" in df.columns else "—")

    st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

    # Row 1
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        sc   = df["sentiment"].value_counts()
        cmap = {"positive":"#4CAF82","negative":"#E05252","neutral":YELLOW,"unknown":"#64748B"}
        fig  = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=0.6,
            marker_colors=[cmap.get(s,"#64748B") for s in sc.index],
            textinfo="label+percent", textfont_size=11,
        ))
        fig.add_annotation(text=f"<b>{pos_pct:.0f}%</b><br>Positive",
                           x=0.5, y=0.5, showarrow=False, font_size=13, font_color=ACCENT)
        fig.update_layout(title="Sentiment Distribution", **PL, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        if "polarity" in df.columns and "rating" in df.columns:
            avg    = df.groupby("rating")["polarity"].mean().reset_index()
            colors = [RED if v < 0 else BLUE for v in avg["polarity"]]
            fig    = go.Figure(go.Bar(
                x=[f"{int(r)} Star{'s' if r>1 else ''}" for r in avg["rating"]],
                y=avg["polarity"].round(3),
                marker_color=colors,
                text=avg["polarity"].round(3),
                textposition="outside",
            ))
            fig.update_layout(title="Average Sentiment vs Rating",
                              xaxis_title="Star Rating", yaxis_title="Avg Polarity", **PL)
            st.plotly_chart(fig, use_container_width=True)

    # Row 2
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        sample = df[df["doc_length"] < 400].sample(min(3000, len(df)), random_state=42)
        scmap  = {"positive":"#4CAF82","negative":"#E05252","neutral":YELLOW,"unknown":"#64748B"}
        fig    = px.scatter(
            sample, x="doc_length",
            y="polarity" if "polarity" in sample.columns else "doc_length",
            color="sentiment", color_discrete_map=scmap, opacity=0.55,
            title="Review Length vs Sentiment",
            labels={"doc_length":"Review Length (Word Count)","polarity":"Polarity Score"},
        )
        fig.update_traces(marker_size=4)
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        if "parent_asin" in df.columns:
            grp = df.groupby("parent_asin").agg(
                n_reviews=("rating","count"),
                title=("title_meta","first") if "title_meta" in df.columns else ("parent_asin","first"),
                pct_positive=("sentiment", lambda x: round((x=="positive").sum()/len(x)*100,1))
            ).sort_values("n_reviews", ascending=False).head(10).reset_index()

            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=["<b>Product</b>","<b># Reviews</b>","<b>% Positive</b>"],
                    fill_color="#1a2235",
                    font=dict(color=ACCENT, size=11, family="Space Mono"),
                    align="left", height=32,
                ),
                cells=dict(
                    values=[
                        grp["title"].apply(lambda x: str(x)[:30]+"…" if len(str(x))>30 else str(x)),
                        grp["n_reviews"].apply(lambda x: f"{x:,}"),
                        grp["pct_positive"].apply(lambda x: f"{x}%"),
                    ],
                    fill_color=[["#111827","#0f172a"]*10],
                    font=dict(color=["#E2E8F0","#E2E8F0",ACCENT], size=10, family="Space Mono"),
                    align="left", height=28,
                )
            )])
            fig.update_layout(title="Top Products by Sentiment", **PL, height=340)
            st.plotly_chart(fig, use_container_width=True)

    # Row 3
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        if "rating" in df.columns:
            rc = df["rating"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=rc.index.astype(str), y=rc.values,
                marker_color=[RED,"#FF8C42",YELLOW,"#7BC67E",ACCENT][:len(rc)],
                text=rc.values, textposition="outside",
            ))
            fig.update_layout(title="Ratings Distribution",
                              xaxis_title="Rating (Stars)", yaxis_title="Number of Reviews", **PL)
            st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        if "polarity" in df.columns and df["polarity"].notna().any():
            fig = px.histogram(
                df.dropna(subset=["polarity"]), x="polarity", nbins=60,
                title="Polarity Score Distribution",
                labels={"polarity":"Polarity Score"},
                color_discrete_sequence=[BLUE],
            )
            fig.add_vline(x=0, line_dash="dash", line_color=RED, line_width=1)
            fig.update_layout(**PL)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Compare":
    if df is None:
        st.info("⬅️  Upload a dataset in the sidebar first.")
        st.stop()

    st.markdown('<div class="section-header">3-Model Comparison · TF-IDF LR  vs  LinearSVC  vs  DistilBERT</div>', unsafe_allow_html=True)

    run_btn = st.button("🚀  RUN ALL 3 MODELS", use_container_width=True)

    if run_btn:
        results = {}
        prog = st.progress(0, text="Training Model 1: TF-IDF + Logistic Regression…")
        try:
            lr_pipe, lr_t, lr_m = train_logistic(df)
            st.session_state["lr_pipe"] = lr_pipe
            st.session_state["lr_t"]    = lr_t
            results["Logistic Regression"] = lr_m
        except Exception as e:
            st.error(f"Model 1 failed: {e}")

        prog.progress(33, text="Training Model 2: TF-IDF + LinearSVC…")
        try:
            svc_pipe, svc_t, svc_m = train_svc(df)
            st.session_state["svc_pipe"] = svc_pipe
            st.session_state["svc_t"]    = svc_t
            results["LinearSVC"] = svc_m
        except Exception as e:
            st.error(f"Model 2 failed: {e}")

        prog.progress(66, text="Loading Model 3: DistilBERT…")
        try:
            hf_pipe = load_distilbert()
            st.session_state["hf_pipe"] = hf_pipe
            bert_m = evaluate_distilbert(df, hf_pipe)
            results["DistilBERT"] = bert_m
            prog.progress(100, text="Done!")
        except Exception as e:
            st.error(f"Model 3 failed: {e}")
            st.warning("DistilBERT requires `transformers` and `torch` in requirements.txt")

        st.session_state["model_results"] = results

    results = st.session_state.get("model_results", {})
    if not results:
        st.stop()

    model_colors = [ACCENT, YELLOW, PURPLE]
    st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

    # Metric cards
    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
    for col, (name, m), color in zip(st.columns(len(results)), results.items(), model_colors):
        with col:
            st.markdown(
                f'<div class="card" style="border-top:3px solid {color};text-align:center">'
                f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{color};margin-bottom:1rem">{name}</div>'
                f'<div class="stat-number" style="color:{color}">{m["f1_macro"]:.4f}</div>'
                f'<div class="stat-label">F1 Macro</div>'
                f'<hr style="border-color:#1f2d45;margin:0.8rem 0">'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-family:Space Mono,monospace;font-size:0.72rem">'
                f'<div><div style="color:#64748b">Accuracy</div><div style="color:#E2E8F0">{m["accuracy"]:.4f}</div></div>'
                f'<div><div style="color:#64748b">Precision</div><div style="color:#E2E8F0">{m["precision"]:.4f}</div></div>'
                f'<div><div style="color:#64748b">Recall</div><div style="color:#E2E8F0">{m["recall"]:.4f}</div></div>'
                f'<div><div style="color:#64748b">Threshold</div><div style="color:#E2E8F0">{m["threshold"]:.2f}</div></div>'
                f'<div><div style="color:#64748b">F1 Pos</div><div style="color:{ACCENT}">{m["f1_pos"]:.4f}</div></div>'
                f'<div><div style="color:#64748b">F1 Neg</div><div style="color:{RED}">{m["f1_neg"]:.4f}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Bar chart comparison
    st.markdown('<div class="section-header">Metric Comparison</div>', unsafe_allow_html=True)
    metric_keys   = ["accuracy","f1_macro","precision","recall"]
    metric_labels = ["Accuracy","F1 Macro","Precision","Recall"]
    fig = go.Figure()
    for (name, m), color in zip(results.items(), model_colors):
        fig.add_trace(go.Bar(
            name=name, x=metric_labels,
            y=[m[k] for k in metric_keys],
            marker_color=color,
            text=[f"{m[k]:.4f}" for k in metric_keys],
            textposition="outside",
        ))
    fig.update_layout(barmode="group", title="Model Metrics Side by Side",
                      yaxis=dict(range=[0,1.1]), **PL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Confusion matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    for col, (name, m), color in zip(st.columns(len(results)), results.items(), model_colors):
        with col:
            cm    = np.array(m["confusion_matrix"])
            total = cm.sum()
            z_text = [[f"{cm[r][c]}<br>({cm[r][c]/total*100:.1f}%)" for c in range(2)] for r in range(2)]
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred: Neg","Pred: Pos"], y=["True: Neg","True: Pos"],
                text=z_text, texttemplate="%{text}",
                colorscale=[[0,"#0A0E1A"],[0.5,"#1a2235"],[1,color]],
                showscale=False,
            ))
            fig.update_layout(title=name, **PL, height=280)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Probability distributions
    st.markdown('<div class="section-header">Positive Probability Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for (name, m), color in zip(results.items(), model_colors):
        fig.add_trace(go.Histogram(x=m["proba"], name=name, opacity=0.6, nbinsx=50, marker_color=color))
    fig.update_layout(barmode="overlay", title="Distribution of Positive-Class Probability",
                      xaxis_title="P(positive)", yaxis_title="Count", **PL)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Classification reports
    st.markdown('<div class="section-header">Full Classification Reports</div>', unsafe_allow_html=True)
    for name, m in results.items():
        with st.expander(f"📋 {name}"):
            st.code(m["report"], language=None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝  Predict":
    st.markdown('<div class="section-header">Live Inference · All 3 Models</div>', unsafe_allow_html=True)

    lr_pipe  = st.session_state.get("lr_pipe")
    svc_pipe = st.session_state.get("svc_pipe")
    hf_pipe  = st.session_state.get("hf_pipe")
    lr_t     = st.session_state.get("lr_t", lr_threshold)
    svc_t    = st.session_state.get("svc_t", 0.5)

    if not all([lr_pipe, svc_pipe, hf_pipe]):
        st.warning("⚠️  Go to **🤖 Model Compare** and click **Run All 3 Models** first.")
        st.stop()

    col_in, col_out = st.columns([1,1], gap="large")
    with col_in:
        st.markdown("**Enter one review per line:**")
        user_text = st.text_area(
            label="input", label_visibility="collapsed",
            placeholder="e.g.\nLove this jacket, fits perfectly!\nTerrible quality, fell apart after one wash.",
            height=240,
        )
        run_btn = st.button("⚡  ANALYZE WITH ALL 3 MODELS", use_container_width=True)

    with col_out:
        if run_btn and user_text.strip():
            lines = [l.strip() for l in user_text.strip().splitlines() if l.strip()]
            with st.spinner("Running all 3 models…"):
                res_list = predict_all(lines, lr_pipe, lr_t, svc_pipe, svc_t, hf_pipe)
            for item in res_list:
                st.markdown(f'<div style="font-size:0.75rem;color:#64748b;font-family:Space Mono,monospace;margin-top:1rem;margin-bottom:0.3rem">"{item["text"][:80]}{"…" if len(item["text"])>80 else ""}"</div>', unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.markdown(result_card(item,"lr","TF-IDF + LR"),  unsafe_allow_html=True)
                with mc2: st.markdown(result_card(item,"svc","LinearSVC"),   unsafe_allow_html=True)
                with mc3: st.markdown(result_card(item,"bert","DistilBERT"), unsafe_allow_html=True)
        elif run_btn:
            st.info("Please enter at least one review.")
        else:
            st.markdown(
                '<div style="color:#334155;font-family:Space Mono,monospace;font-size:0.8rem;padding-top:3rem;text-align:center">'
                '← type a review and hit ANALYZE<span class="blink">_</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-header">Quick Demo Examples</div>', unsafe_allow_html=True)
    demos = [
        ("Positive 🟢", "Absolutely love this bag! Great quality stitching and the color is stunning. Would buy again."),
        ("Negative 🔴", "Complete waste of money. The zipper broke after a week and the material feels cheap."),
        ("Mixed 🟡",    "It looks okay in photos but the actual item is smaller than expected. Shipping was fast though."),
    ]
    for col, (label, text) in zip(st.columns(3), demos):
        with col:
            if st.button(label, use_container_width=True, key=f"demo_{label}"):
                with st.spinner("Running…"):
                    res = predict_all([text], lr_pipe, lr_t, svc_pipe, svc_t, hf_pipe)
                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.markdown(result_card(res[0],"lr","TF-IDF + LR"),  unsafe_allow_html=True)
                with mc2: st.markdown(result_card(res[0],"svc","LinearSVC"),   unsafe_allow_html=True)
                with mc3: st.markdown(result_card(res[0],"bert","DistilBERT"), unsafe_allow_html=True)
