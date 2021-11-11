"""Plots top n hashtags in dataset"""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from utils import open_json_as_dataframe


keyword = "mentions"
symbol = "@"

df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics_4.pkl")
df = df[[keyword, "sentiment_status"]]
df = df.dropna(subset=[keyword])

# Extract most frequency hashtags
num_hts = 20
ht_list_not_flat = [(ht_list, status) for ht_list, status in zip(df[keyword], df["sentiment_status"])]
hts = [f"{symbol}{ht.lower()}" for ht_list in ht_list_not_flat for ht in ht_list[0]]
ht_freq = {ht: hts.count(ht) for ht in hts}
ht_counter = Counter(ht_freq)
top_hts = ht_counter.most_common(num_hts)

neut_top_hts = []
for top_ht, count in top_hts:
    count_neut = len([1 for x in ht_list_not_flat if top_ht[1:] in [ht.lower() for ht in x[0]] and x[1] == "neutral"])
    neut_top_hts.append((top_ht, count_neut))
pos_top_hts = []
for top_ht, count in top_hts:
    count_pos = len([1 for x in ht_list_not_flat if top_ht[1:] in [ht.lower() for ht in x[0]] and x[1] == "positive"])
    pos_top_hts.append((top_ht, count_pos))
neg_top_hts = []
for top_ht, count in top_hts:
    count_neg = len([1 for x in ht_list_not_flat if top_ht[1:] in [ht.lower() for ht in x[0]] and x[1] == "negative"])
    neg_top_hts.append((top_ht, count_neg))

pos_dist = []
for neut, pos, neg, in zip(neut_top_hts, pos_top_hts, neg_top_hts):
    ht = neut[0]
    per = 100*pos[1]/(pos[1] + neut[1] + neg[1])
    pos_dist.append((ht, per))

fig, ax = plt.subplots()
bar1 = ax.barh([ht for ht, freq in pos_top_hts], [freq for ht, freq in pos_top_hts], color="olivedrab", label="Positive Sentiment")
bar2 = ax.barh([ht for ht, freq in pos_top_hts], [freq for ht, freq in neut_top_hts], color="gold", left=[freq for ht, freq in pos_top_hts], label="Neutral Sentiment")
left_2 = [tup_1[1] + tup_2[1] for tup_1, tup_2 in zip(neut_top_hts, pos_top_hts)]
bar3 = ax.barh([ht for ht, freq in pos_top_hts], [freq for ht, freq in neg_top_hts], color="indianred", left=left_2, label="Negative Sentiment")
ax.set_ylabel("Top Hashtags")
ax.set_xlabel("Number of Tweets")
plt.legend()

ax2 = ax.twiny()
lns1 = ax2.plot([pd[1] for pd in pos_dist], [ht for ht, freq in pos_top_hts], label="Positive Proportion (%)", color="slategrey", marker='o', markersize=3, linestyle="dashed", linewidth=1)
ax2.set_xlim(0, 105)
ax2.set_xlabel("Proportion of Tweets with Positive Sentiment (%)")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
