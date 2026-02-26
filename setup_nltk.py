"""
setup_nltk.py
Run once before launching the app if TextBlob corpora aren't installed.
On Streamlit Cloud this runs automatically via packages.txt / startup.
"""
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

PACKAGES = [
    "punkt",
    "averaged_perceptron_tagger",
    "opinion_lexicon",
    "stopwords",
    "wordnet",
]

for pkg in PACKAGES:
    try:
        nltk.download(pkg, quiet=True)
        print(f"  ✓ {pkg}")
    except Exception as e:
        print(f"  ✗ {pkg}: {e}")

# TextBlob corpora
try:
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "textblob.download_corpora"],
        check=True, capture_output=True
    )
    print("  ✓ textblob corpora")
except Exception as e:
    print(f"  ✗ textblob corpora: {e}")

print("Done.")
