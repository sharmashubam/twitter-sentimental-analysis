import streamlit as st
import snscrape.modules.twitter as sntwitter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
import warnings


def scrape_tweets(username, tweet_count):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterUserScraper(username).get_items()):
        if i+1 > tweet_count:
            break
        tweets.append(tweet)
    return tweets


def map_label(label):
    if label == "LABEL_0":
        return "negative"
    elif label == "LABEL_1":
        return "positive"
    else:
        return None


def analyze_sentiment(username, tweet_count):
    with st.spinner("Analyzing tweets..."):
        tweets = scrape_tweets(username, tweet_count)
        print(f"Total number of scraped tweets: {len(tweets)}")
        tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased")
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        labels = []
        for tweet in tweets:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                result = sentiment_analyzer(tweet.content)
                print(f"Tweet: {tweet.content}")
                label = map_label(result[0]['label'])
                if label:
                    print(f"Sentiment: {label}")
                    labels.append(label)

        sentiment_counts = {
            'Positive': labels.count('positive'),
            'Negative': labels.count('negative')
        }

        print(f"Sentiment Counts: {sentiment_counts}")

        x = sentiment_counts.keys()
        y = sentiment_counts.values()

        colors = ['green', 'red']

        cmap = plt.cm.colors.ListedColormap(colors)

        fig, ax = plt.subplots()
        ax.bar(x, y, color=cmap(np.arange(len(y))))

        ax.set_title('Tweets Analysis')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')

        st.pyplot(fig)


def main():
    st.title("TweetAlysis")
    username = st.text_input("Enter your Twitter username", key="username_input")
    tweet_count = st.number_input("Enter the number of tweets to analyze", min_value=1, max_value=500, value=50,
                                  key="tweet_count_input")

    if st.button("Do Analysis", key="sentiment_button"):
        if username:
            analyze_sentiment(username, tweet_count)
        else:
            st.warning("Please enter a Twitter username.")
    else:
        st.info("Click the 'Do Analysis' button to analyze tweets.")


if __name__ == "__main__":
    main()
