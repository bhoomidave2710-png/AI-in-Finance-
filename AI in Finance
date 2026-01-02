import streamlit as st
import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(page_title="Social Media Big Data Analyzer", layout="wide")

st.title("📊 Social Media Big Data Analyzer")
st.write("Analyze Reddit trending topics using NLP, TF-IDF and Visualization")

@st.cache_data
def load_data():
    url = "https://www.reddit.com/r/popular/.rss"
    feed = feedparser.parse(url)
    texts = [entry.title for entry in feed.entries]
    return pd.DataFrame(texts, columns=["text"])

df = load_data()

st.subheader("Trending Reddit Data")
st.dataframe(df.head(10))

vectorizer = TfidfVectorizer(stop_words="english", max_features=30)
tfidf_matrix = vectorizer.fit_transform(df["text"])
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

scores_only = tfidf_df.sum(axis=0).sort_values(ascending=False)

st.subheader("TF-IDF Scores")
st.dataframe(
    scores_only.reset_index().rename(
        columns={"index": "Word", 0: "TF-IDF Score"}
    )
)

topic_name = st.text_input("Enter Topic Name", "Reddit Trending Analysis").title()

def suggest_category(words):
    ai = {"ai", "chatgpt", "openai", "machine", "artificial"}
    finance = {"bitcoin", "crypto", "stock", "market", "economy"}
    sports = {"cricket", "football", "match", "olympics"}
    politics = {"election", "government", "minister", "policy"}
    tech = {"apple", "google", "microsoft", "tesla"}

    for w in words:
        if w in ai:
            return "Artificial Intelligence"
        if w in finance:
            return "Finance & Crypto"
        if w in sports:
            return "Sports"
        if w in politics:
            return "Politics"
        if w in tech:
            return "Technology"

    return "General Trending Topic"

category = suggest_category(scores_only.index[:10])
st.success(f"Detected Trend Category: {category}")

combined_text = " ".join(df["text"])
wordcloud = WordCloud(width=900, height=450, background_color="white").generate(combined_text)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
ax.set_title(f"{topic_name} ({category})")
st.pyplot(fig)

all_words = combined_text.lower().split()
word_counts = Counter(all_words)

freq_df = pd.DataFrame(
    word_counts.items(),
    columns=["Word", "Frequency"]
).sort_values(by="Frequency", ascending=False)

st.subheader("Repeated Words and Frequency")
st.dataframe(freq_df.head(20))

search_word = st.text_input("Enter a word to check TF-IDF score").lower()

if search_word:
    if search_word in scores_only.index:
        st.info(f"TF-IDF Score of '{search_word}': {scores_only[search_word]:.4f}")
    else:
        st.warning("Word not found in trending data")
streamlit
feedparser
pandas
scikit-learn
wordcloud
matplotlib
social-media-big-data-analyzer/
│
├── app.py
├── requirements.txt
└── README.md
Social Media Big Data Analyzer

This Streamlit application analyzes Reddit trending topics using
TF-IDF, NLP, and data visualization.

Run the app:
streamlit run app.py
pip install -r requirements.txt
streamlit run app.py
