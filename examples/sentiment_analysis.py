"""This plot runs a sentiment analysis on two inputs and plots results,
detailing polarity and subjectivity."""
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
import tweepy
from utils import clean, open_json_as_dataframe

# Import standard df.
df_1, meta = open_json_as_dataframe("../data/ice_ban-23-Jan-2021.json")
df_2, meta = open_json_as_dataframe("../data/ev-24-Jan-2021.json")
df_3, meta = open_json_as_dataframe("../data/electric_car_uk-24-Jan-2021.json")
df_4, meta = open_json_as_dataframe("../data/electric_vehicle_uk-23-Jan-2021.json")
df = pd.concat([df_1, df_2, df_3, df_4])

favourites = df.favorite_count.to_list()
retweets = df.retweet_count.to_list()

tweet_list = df.text.to_list()
tweet_list = clean(tweet_list)

tweet_polarity = [TextBlob(tweet).polarity for tweet in tweet_list]
tweet_subjectivity = [TextBlob(tweet).sentiment.subjectivity for tweet in tweet_list]
df["polarity"] = tweet_polarity
df["subjectivity"] = tweet_subjectivity

neutral_threshold = 0
tweet_num_positive = sum(pol > neutral_threshold for pol in tweet_polarity)
tweet_num_negative = sum(pol < -neutral_threshold for pol in tweet_polarity)
tweet_num_neutral = len(tweet_polarity) - tweet_num_positive - tweet_num_negative

cm = plt.cm.get_cmap('GnBu')
sc = plt.scatter(tweet_polarity, tweet_subjectivity, s=favourites, alpha=0.65,
                 c=retweets, cmap=cm, vmin=-18, vmax=50)
plt.xlabel("Polarity (Positive v Negative)")
plt.ylabel("Subjectivity (How Emotive?)")
plt.xlim(-1.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.show()


labels = ['Positive', 'Neutral', 'Negative']
sizes = [tweet_num_positive, tweet_num_neutral, tweet_num_negative]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
