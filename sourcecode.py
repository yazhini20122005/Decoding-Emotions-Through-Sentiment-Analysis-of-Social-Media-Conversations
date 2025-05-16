import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

st.set_page_config(layout="wide")

# Load dataset
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, header=None, names=['ID', 'Entity', 'Sentiment', 'Tweet'], quotechar='"', skipinitialspace=True)
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

# Analyze sentiment distribution
def analyze_sentiment(df):
    return df.groupby(['Entity', 'Sentiment']).size().unstack(fill_value=0)

# Generate word cloud data
def generate_word_cloud_data(tweets, sentiment):
    words = []
    for tweet in tweets[tweets['Sentiment'] == sentiment]['Tweet']:
        if isinstance(tweet, str):
            cleaned = re.sub(r"[^a-zA-Z\s']", '', tweet.lower())
            words.extend([word for word in cleaned.split() if len(word) > 3])
    return dict(Counter(words).most_common(50))

# Bar chart
def plot_bar_chart(sentiment_counts):
    st.subheader("Sentiment Distribution by Entity")
    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336', '#2196F3', '#FF9800'])
    ax.set_xlabel("Entity")
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Sentiment Distribution by Entity")
    ax.legend(title="Sentiment")
    st.pyplot(fig)

# Pie charts
def plot_pie_charts(sentiment_counts):
    st.subheader("Sentiment Proportions by Entity")
    for entity in sentiment_counts.index:
        fig, ax = plt.subplots()
        counts = sentiment_counts.loc[entity]
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
               colors=['#4CAF50', '#F44336', '#2196F3', '#FF9800'])
        ax.set_title(f'Sentiment Proportions for {entity}')
        st.pyplot(fig)

# Word clouds
def plot_word_clouds(df):
    st.subheader("Word Clouds for 'Borderlands'")
    for sentiment in ['Positive', 'Negative']:
        words = generate_word_cloud_data(df[df['Entity'] == 'Borderlands'], sentiment)
        if words:
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud: {sentiment} Borderlands Tweets')
            st.pyplot(fig)

# Streamlit UI
def main():
    st.title("📊 Sentiment Analysis of Tweets")
    uploaded_file = st.file_uploader("Upload your CSV file (twitter_training.csv)", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            df = clean_data(df)
            st.success("Data loaded and cleaned successfully.")
            st.write("Sample Data:", df.head())

            sentiment_counts = analyze_sentiment(df)
            st.write("Sentiment Distribution Table:", sentiment_counts)

            if 'Borderlands' in sentiment_counts.index:
                positive_pct = sentiment_counts.loc['Borderlands', 'Positive'] / sentiment_counts.loc['Borderlands'].sum() * 100
                st.info(f"📌 Interesting Fact: {positive_pct:.1f}% of Borderlands tweets are Positive, reflecting strong fan enthusiasm.")

            plot_bar_chart(sentiment_counts)
            plot_pie_charts(sentiment_counts)
            plot_word_clouds(df)

if __name__ == "__main__":
    main()
