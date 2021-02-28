"""Adds sentiment scores to data model and saves in processed data directory."""
import pandas as pd
from textblob import TextBlob

from utils import clean, open_json_as_dataframe

dfs_1 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-31-Jan-2021.json")[0] for index in range(1, 5)]
dfs_2 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-01-Feb-2021.json")[0] for index in range(5, 10)]
df = pd.concat(dfs_1 + dfs_2)

tweet_list = df.text.to_list()
tweet_list = clean(tweet_list)

tweet_polarity = [TextBlob(tweet).polarity for tweet in tweet_list]
tweet_subjectivity = [TextBlob(tweet).sentiment.subjectivity for tweet in tweet_list]
df["polarity"] = tweet_polarity
df["subjectivity"] = tweet_subjectivity
df.to_pickle("../processed_data/ICE_ban_November_2020_sent.pkl")  # Save as pickle
