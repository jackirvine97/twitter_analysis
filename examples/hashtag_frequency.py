"""Plots top n hashtags in dataset"""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from utils import open_json_as_dataframe

dfs_1 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-31-Jan-2021.json")[0] for index in range(1, 5)]
dfs_2 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-01-Feb-2021.json")[0] for index in range(5, 10)]
df = pd.concat(dfs_1 + dfs_2)

# Extract most frequency hashtags
num_hts = 17
ht_list_not_flat = [ht_list for ht_list in df.hashtags]
hts = [f"#{ht.lower()}" for ht_list in ht_list_not_flat for ht in ht_list]
ht_freq = {ht: hts.count(ht) for ht in hts}
ht_counter = Counter(ht_freq)
top_hts = ht_counter.most_common(num_hts)

plt.figure()
plt.bar([ht for ht, freq in top_hts], [freq for ht, freq in top_hts])
plt.xticks(rotation=90)
plt.xlabel("Top Hashtags")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.show()
