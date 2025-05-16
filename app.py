import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter

st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

# Load dataset directly
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, header=None, names=['ID', 'Entity', 'Sentiment', 'Tweet'], quotechar='"', skipinitialspace=True)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# Clean dataset
def clean_data(df):
    df = df.dropna(subset=['ID', 'Entity', 'Sentiment', 'Tweet'])
    df = df[df['Tweet'].str.strip() != '']
    valid_sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    df['Sentiment'] = df['Sentiment'].str.capitalize()
    df = df[df['Sentiment'].isin(valid_sentiments)]
    df['Entity'] = df['Entity'].str.strip()
    df['Tweet'] = df['Tweet'].str.strip()
    return df

# Sentiment distribution
def analyze_sentiment(df):
    return df.groupby(['Entity', 'Sentiment']).size().unstack(fill_value=0)

# Word frequency
def generate_word_cloud_data(tweets, sentiment):
    words = []
    for tweet in tweets[tweets['Sentiment'] == sentiment]['Tweet']:
        if isinstance(tweet, str):
            cleaned = re.sub(r"[^a-zA-Z\s']", '', tweet.lower())
            words.extend([word for word in cleaned.split() if len(word) > 3])
    return dict(Counter(words).most_common(50))

# App logic
def main():
    st.title("📊 Twitter Sentiment Analyzer")

    # Fixed file path (replace with your file location if needed)
    file_path = "twitter_training.csv"
    df = load_data(file_path)

    if df is not None:
        df = clean_data(df)

        if df.empty:
            st.warning("No valid data found after cleaning.")
            return

        st.success("✅ Data loaded and cleaned successfully!")
        st.dataframe(df.head(10))

        st.markdown("## 🔍 Choose Analysis Type")
        analysis_option = st.selectbox("Choose analysis type:", [
            "Sentiment Distribution",
            "Pie Chart by Entity",
            "Word Cloud by Sentiment"
        ])

        sentiment_counts = analyze_sentiment(df)

        if analysis_option == "Sentiment Distribution":
            st.subheader("📊 Sentiment Distribution by Entity")
            st.bar_chart(sentiment_counts)

        elif analysis_option == "Pie Chart by Entity":
            selected_entity = st.selectbox("Select an Entity", sentiment_counts.index)
            entity_data = sentiment_counts.loc[selected_entity]

            fig, ax = plt.subplots()
            ax.pie(entity_data, labels=entity_data.index, autopct='%1.1f%%',
                   colors=['#4CAF50', '#F44336', '#2196F3', '#FF9800'])
            ax.set_title(f"Sentiment Breakdown: {selected_entity}")
            st.pyplot(fig)

        elif analysis_option == "Word Cloud by Sentiment":
            selected_entity = st.selectbox("Select Entity for Word Cloud", df['Entity'].unique())
            selected_sentiment = st.radio("Choose Sentiment", ['Positive', 'Negative'])

            filtered_df = df[df['Entity'] == selected_entity]
            word_data = generate_word_cloud_data(filtered_df, selected_sentiment)

            if word_data:
                wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_data)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f"{selected_sentiment} Word Cloud for {selected_entity}")
                st.pyplot(fig)
            else:
                st.warning("No matching words found.")

if __name__ == "__main__":
    main()

