"""Plots top n mentions in dataset"""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from utils import open_json_as_dataframe

# Import standard df.
df, meta = open_json_as_dataframe("../data/test-24-Jan-2021.json")

# Extract most frequency hasmentionags
num_mentions = 17
mention_list_not_flat = [mention_list for mention_list in df.mentions]
mentions = [f"@{mention.lower()}" for mention_list in mention_list_not_flat for mention in mention_list]
mention_freq = {mention: mentions.count(mention) for mention in mentions}
mention_counter = Counter(mention_freq)
top_mentions = mention_counter.most_common(num_mentions)

plt.figure()
plt.bar([mention for mention, freq in top_mentions], [freq for mention, freq in top_mentions])
plt.xticks(rotation=90)
plt.xlabel("Top Handles")
plt.ylabel("Number of Mentions")
plt.tight_layout()
plt.show()
