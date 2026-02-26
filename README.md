# 🔮 SentimentOS — Amazon Fashion Sentiment Analyzer

A dark-mode Streamlit app for sentiment analysis on Amazon Fashion reviews, powered by a TF-IDF + Logistic Regression pipeline.

---

## Features

- **Predict Mode** — type any review text (one per line) and get instant sentiment + confidence
- **EDA Mode** — upload your raw JSONL files for full exploratory analysis (ratings, polarity, token distributions, top products)
- **Train Mode** — train the model on your uploaded dataset right inside the browser
- **Batch Mode** — upload a CSV/JSONL, run bulk prediction, download results

---

## Project Structure

```
sentiment_app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level deps (Streamlit Cloud)
├── setup_nltk.py           # Run once to download NLTK corpora
├── .gitignore
├── .streamlit/
│   └── config.toml         # Theme & server config
├── assets/
│   └── style.css           # Custom dark-mode stylesheet
└── model/
    ├── __init__.py
    └── sentiment_model.py  # Training, loading, and prediction logic
```

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python setup_nltk.py
```

### 5. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> **Do NOT push your `.jsonl` data files** — they are gitignored. You upload them in-browser via the app's file uploader.

### Step 2 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repo, branch (`main`), and set the **Main file path** to `app.py`.
4. Click **Deploy**.

Streamlit Cloud will automatically:
- Install everything in `requirements.txt`
- Install system packages in `packages.txt`
- Serve your app at `https://<your-app>.streamlit.app`

### Step 3 — Use the app

1. Open the live URL.
2. Go to **EDA & Batch → Exploratory Analysis** and upload your two JSONL files.
3. Go to **Train Model** and click **Start Training**.
4. Switch to **Predict** to run live inference.

---

## Model Details

| Component | Details |
|-----------|---------|
| Vectorizer | TF-IDF, max 100k features, ngram (1,2), min_df=2 |
| Classifier | Logistic Regression, saga solver, balanced class weight |
| Labels | `rating ≥ 4` → positive, `rating ≤ 2` → negative (3 excluded) |
| Threshold | Swept 0.10–0.90, optimised for macro-F1 on validation |
| Persistence | Pickled to `model/trained_model.pkl` |

---

## Notes

- The trained model pickle is gitignored — users re-train in-browser on their own data.
- For production use, consider replacing the pickle with `joblib` and adding model versioning.
- TextBlob polarity runs on a 20k-row sample for speed; adjust `sample_n` in `app.py` if needed.
