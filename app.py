import streamlit as st
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
wordcloud = WordCloud(
width=900,
height=450,
background_color="white"
).generate(combined_text)


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
