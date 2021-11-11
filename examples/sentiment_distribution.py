import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import clean, open_json_as_dataframe

df = pd.read_pickle("../processed_data/ICE_Ban_November_2020_sent_topics_15 3.pkl")

analyser = SentimentIntensityAnalyzer()

pol = [analyser.polarity_scores(tweet)["compound"] for tweet in df.text]
df["vd_polarity"] = pol

df = df.loc[df["lang"] == "en"]  # Filter out languages.
df = df.loc[df["rt"] == False]  # Filter out languages.
fig, ax = plt.subplots(figsize=(9, 6))

conditions = [
    (df['dominant_topic'] == 0),
    (df['dominant_topic'] == 1),
    (df['dominant_topic'] == 2),
    (df['dominant_topic'] == 3),
    (df['dominant_topic'] == 4),
    (df['dominant_topic'] == 5),
    (df['dominant_topic'] == 6),
    (df['dominant_topic'] == 7),
    (df['dominant_topic'] == 8),
    (df['dominant_topic'] == 9),
    (df['dominant_topic'] == 10),
    (df['dominant_topic'] == 11),
    (df['dominant_topic'] == 12),
    (df['dominant_topic'] == 13),
    (df['dominant_topic'] == 14)
]

choices = ['ICE Ban', 'ICE Ban', 'ICE Ban', "10pt Plan", "Van", "ICE Ban", "Petrol Sales", "Wind Power", "ICE Ban", "Green Tech", "PC Impact", "ICE Ban", "ICE Ban", "Wind Power", "10pt Plan"]
choices = ['ICE Ban', 'P Sales', 'Tech', 'ICE Ban', 'ICE Ban', 'Wind','ICE Ban', 'EVs', 'ICE Ban', 'ICE Ban', 'pc impact', 'ICE Ban', 'ICE Ban', 'Net Zero', '10pt Plan']
df['topic'] = np.select(conditions, choices, default='black')


sns.violinplot(x=df['topic'], y=df['polarity'], scale="count", bw="scott", linewidth=1, width=1.05)
ax.set_title('Topic Sentiment Distribution', fontsize=11, loc='center')
ax.set_xlabel('Topics', fontsize=11)
ax.set_ylabel('Sentiment Score', fontsize=11)

plt.tick_params(axis='y', which='major', labelsize=11)
plt.tick_params(axis='x', which='major', labelsize=11, rotation=90)
ax.yaxis.tick_left()  # where the y axis marks will be
plt.tight_layout()
plt.show()

"""Below is WIP code for magnitude topic sentiment bar charts."""


# ff = df[["inferred_topic", "status"]]

# topic_list = ff.inferred_topic.to_list()
# status_list = ff.status.to_list()
# d = {}
# for topic in topic_list:
#     d[topic] = [0, 0, 0]

# for topic, status in zip(topic_list, status_list):
#     if status == "pos":
#         d[topic][0] += 1
#     if status == "neutral":
#         d[topic][1] += 1
#     if status == "neg":
#         d[topic][2] += 1

# for i, v in d.items():
#     print(i, v)


# import numpy as np
# import matplotlib.pyplot as plt


# N = 5
# Pos = (1140, 287, 372, 943, 1055)
# Neut = (572, 140, 248, 332, 7)
# Neg = (308, 65, 32, 165, 1)
# ind = np.arange(N)
# width = 0.35

# p1 = plt.bar(ind, Pos, width, color='olivedrab')
# p2 = plt.bar(ind, Neut, width, color='orange')
# p3 = plt.bar(ind, Neg, width, color='firebrick')

# plt.ylabel('Number of Tweets')
# plt.xticks(ind, ('EV General', 'ICE Ban', 'Battery Technology', 'Charging', 'Nissan Sunderland'))
# plt.legend((p1[0], p2[0], p3[0]), ('Positive', 'Neutral', 'Negative'))

# plt.show()
# plt.show()
