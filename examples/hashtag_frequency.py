"""Plots top n hashtags in dataset"""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from utils import open_json_as_dataframe

# Import standard df.
df_1, meta = open_json_as_dataframe("../data/ice_ban-23-Jan-2021.json")
df_2, meta = open_json_as_dataframe("../data/ev-24-Jan-2021.json")
df_3, meta = open_json_as_dataframe("../data/electric_car_uk-24-Jan-2021.json")
df_4, meta = open_json_as_dataframe("../data/electric_vehicle_uk-23-Jan-2021.json")
df = pd.concat([df_1, df_2, df_3, df_4])

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
