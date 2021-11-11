"""This script generates a star shaped wordcloud."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import re
import time
import tweepy
from utils import (open_json_as_dataframe, search_past_7_days)
from wordcloud import STOPWORDS, WordCloud


dfs_1 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-31-Jan-2021.json")[0] for index in range(1, 5)]
dfs_2 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-01-Feb-2021.json")[0] for index in range(5, 10)]
df = pd.concat(dfs_1 + dfs_2)
df, meta = open_json_as_dataframe("../data/EVs_15th_January_2020-28-Feb-2021.json")
df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics.pkl")

tweets_text = df.text.to_list()
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

# Uncomment to load in mask.
# mask = np.array(Image.open('star.png'))

wordcloud = WordCloud(
    background_color="white",
    width=1600, height=800,
    max_words=200,
    # mask=mask,
)
wordcloud.generate(filtered_string)

print(f"Number of tweets processed: {tweet_count}")

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
