"""This script generates a star shaped wordcloud."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
import time
import tweepy
from wordcloud import STOPWORDS, WordCloud

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

search_term = ["pret a manger"]
number_of_tweets = 2000

# Collect tweets.
tweets = tweepy.Cursor(api.search, q=search_term, lang="en").items(number_of_tweets)
tweets_text = [tweet.text for tweet in tweets]
tweet_count = len(tweets_text)

# Combine and filter.
tweets_string = ''.join(tweets_text)
no_links = re.sub(r'http\S+', '', tweets_string)
no_unicode = re.sub(r"\\[a-z][a-z]?[0-9]+", '', no_links)
no_special_characters = re.sub('[^A-Za-z ]+', '', no_unicode)

# Divide, ignore single characters and remove stopwords.
words = no_special_characters.split(" ")
words = [w for w in words if len(w) > 2]
words = [w.lower() for w in words]
STOPWORDS.update(["pret", "manger", "pret a manger", "retweet", "pretamanger",
                  "dr", "janaway", "dr janaway", "drjanaway", "morei", "sissy",
                  "chefsay", "ferdinand"])
words = [w for w in words if w not in STOPWORDS]
filtered_string = ','.join(words)

mask = np.array(Image.open('star.png'))

wordcloud = WordCloud(
    background_color="white",
    width=1600, height=800,
    max_words=200,
    mask=mask,
)
wordcloud.generate(filtered_string)

print(f"Number of tweets processed: {tweet_count}")

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
