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
strong_threshold = 0.5
tweet_num_strong_positive = sum(pol > strong_threshold for pol in tweet_polarity)
tweet_num_strong_negative = sum(pol < -strong_threshold for pol in tweet_polarity)
tweet_num_weak_positive = sum((pol > neutral_threshold) and (pol <= strong_threshold) for pol in tweet_polarity)
tweet_num_weak_negative = sum((pol < -neutral_threshold) and (pol >= -strong_threshold) for pol in tweet_polarity)
tweet_num_neutral = (len(tweet_polarity) - tweet_num_strong_positive
                     - tweet_num_strong_negative - tweet_num_weak_positive
                     - tweet_num_weak_negative)

# Plot scatter map.
cm = plt.cm.get_cmap('GnBu')
sc = plt.scatter(tweet_polarity, tweet_subjectivity, s=favourites, alpha=0.65,
                 c=retweets, cmap=cm, vmin=-18, vmax=50)
plt.xlabel("Polarity (Positive v Negative)")
plt.ylabel("Subjectivity (How Emotive?)")
plt.xlim(-1.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.show()

# Plot pie chart.
labels = ['Strong Positive', 'Weak Positive', 'Neutral', 'Weak Negative',
          'Strong Negative']
sizes = [tweet_num_strong_positive, tweet_num_weak_positive, tweet_num_neutral,
         tweet_num_weak_negative, tweet_num_strong_negative]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

outside_values = [tweet_num_strong_positive + tweet_num_weak_positive, tweet_num_neutral,
                  tweet_num_weak_negative + tweet_num_strong_negative]
outside_names = ['Positive',  'Neutral',  'Negative']
inside_values = [tweet_num_strong_positive, tweet_num_weak_positive, tweet_num_neutral,
                 tweet_num_strong_negative, tweet_num_weak_negative]
inside_names = ['Strong', 'Weak', '', 'Strong', 'Weak']

# Plot ring chart.
a, b, c = [plt.cm.Greens, plt.cm.Greys, plt.cm.Reds]
# First Ring (outside)
fig, ax = plt.subplots(figsize=(7, 7))
ax.axis('equal')
mypie, _, _ = ax.pie(outside_values, radius=1.3, labels=outside_names,
                     textprops={'color': "black", "size": 10, "alpha": 0.84},
                     colors=[a(0.7), b(0.4), c(0.7)], autopct='%.1f%%',
                     pctdistance=0.89)
plt.setp(mypie, width=0.3, edgecolor='white')

# Second Ring (Inside)
mypie2, _, _ = ax.pie(inside_values, radius=1.3-0.3, labels=inside_names,
                      textprops={'color': "black", "size": 10, "alpha": 0.84},
                      labeldistance=0.4, autopct="%.1f%%", pctdistance=0.8,
                      colors=[a(0.6), a(0.3), b(0.2), c(0.6), c(0.4)])
plt.setp(mypie2, width=0.4, edgecolor='white')
plt.margins(0, 0)
# Central String
ax.text(0., 0., f"{len(tweet_list)}\nTotal Tweets", horizontalalignment='center',
        verticalalignment='center', size=11, weight='bold')
plt.show()
