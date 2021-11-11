"""Plots top n mentions in dataset"""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from utils import open_json_as_dataframe

# Import standard df.
dfs_1 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-31-Jan-2021.json")[0] for index in range(1, 5)]
dfs_2 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-01-Feb-2021.json")[0] for index in range(5, 10)]
df = pd.concat(dfs_1 + dfs_2)
# df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics.pkl")
# **** Load in EV 2020 trends ****
months_yy = ["January_2020", "February_2020", "March_2020", "April_2020", "May_2020", "June_2020", "July_2020",
             "August_2020", "September_2020", "October_2020", "November_2020", "December_2020", "January_2021"]


# **** Load in EV 2020 trends ****
months_yy = ["January_2020", "February_2020", "March_2020", "April_2020", "May_2020", "June_2020", "July_2020",
             "August_2020", "September_2020", "October_2020", "November_2020", "December_2020", "January_2021"]

dfs_1 = [open_json_as_dataframe(f"../data/EVs_15th_{month_yy}-02-Mar-2021.json")[0] for month_yy in months_yy]
dfs_2 = [open_json_as_dataframe(f"../data/EVs_28th_{month_yy}-02-Mar-2021.json")[0] for month_yy in months_yy]
dfs_3 = [open_json_as_dataframe(f"../data/EVs_15th_January_2_2020-02-Mar-2021.json")[0]]
dfs_4 = [open_json_as_dataframe(f"../data/EVs_15th_January_3_2020-02-Mar-2021.json")[0]]
dfs_5 = [open_json_as_dataframe(f"../data/EVs_15th_Feb_2021-03-Mar-2021.json")[0]]
dfs_6 = [open_json_as_dataframe(f"../data/EV-Mar2021-07-Mar-2021.json")[0]]
dfs_7 = [open_json_as_dataframe(f"../data/EVs_28th_Feb_2021-07-Mar-2021.json")[0]]
# dfs_8 = [open_json_as_dataframe(f"../data/electric_car_uk-24-Jan-2021.json")[0]]
# dfs_9 = [open_json_as_dataframe(f"../data/electric_vehicle_uk-23-Jan-2021.json")[0]]
df = pd.concat(dfs_1 + dfs_2 + dfs_3 + dfs_4 + dfs_5 + dfs_6 + dfs_7)


# df = df.loc[df["rt"]==False]  # Filter out RTs
df = df[~df['text'].str.contains('hr6201')]
df = df[~df['text'].str.contains('HR6201')]
df = df[~df['text'].str.contains('JohnKingCNN')]
df = df[~df['text'].str.contains('Started a space company &gt')]
df = df[~df['text'].str.contains('CraigCaplan')]
df = df[~df['text'].str.contains('Shane_Evs')]
df = df[~df['text'].str.contains('Evs_Dubai')]

# Extract most frequency hasmentionags
num_mentions = 20
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
