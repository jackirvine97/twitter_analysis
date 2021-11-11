"""Adds sentiment scores to data model and saves in processed data directory."""
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import clean, open_json_as_dataframe


# **** Load in EV 2020 trends ****

# months_yy = ["January_2020", "February_2020", "March_2020", "April_2020", "May_2020", "June_2020", "July_2020",
#              "August_2020", "September_2020", "October_2020", "November_2020", "December_2020", "January_2021"]

# dfs_1 = [open_json_as_dataframe(f"../data/EVs_15th_{month_yy}-02-Mar-2021.json")[0] for month_yy in months_yy]
# dfs_2 = [open_json_as_dataframe(f"../data/EVs_28th_{month_yy}-02-Mar-2021.json")[0] for month_yy in months_yy]
# dfs_3 = [open_json_as_dataframe(f"../data/EVs_15th_January_2_2020-02-Mar-2021.json")[0]]
# dfs_4 = [open_json_as_dataframe(f"../data/EVs_15th_January_3_2020-02-Mar-2021.json")[0]]
# dfs_5 = [open_json_as_dataframe(f"../data/EVs_15th_Feb_2021-03-Mar-2021.json")[0]]
# dfs_6 = [open_json_as_dataframe(f"../data/EV-Mar2021-07-Mar-2021.json")[0]]
# dfs_7 = [open_json_as_dataframe(f"../data/EVs_28th_Feb_2021-07-Mar-2021.json")[0]]
# dfs_8 = [open_json_as_dataframe(f"../data/electric_car_uk-24-Jan-2021.json")[0]]
# dfs_9 = [open_json_as_dataframe(f"../data/electric_vehicle_uk-23-Jan-2021.json")[0]]
# dfs_10 = [open_json_as_dataframe(f"../data/EV-Mar-12_2020_4500-21-Mar-2021.json")[0]]
# dfs_11 = [open_json_as_dataframe(f"../data/EV-Mar-28-2021-27-Mar-2021.json")[0]]
# df = pd.concat(dfs_1 + dfs_2 + dfs_3 + dfs_4 + dfs_5 + dfs_6 + dfs_7 + dfs_8 + dfs_9 + dfs_10 + dfs_11)

# df = df[~df['text'].str.contains('hr6201')]
# df = df[~df['text'].str.contains('HR6201')]
# df = df[~df['text'].str.contains('JohnKingCNN')]
# df = df[~df['text'].str.contains('Started a space company &gt')]
# df = df[~df['text'].str.contains('CraigCaplan')]
# df = df[~df['text'].str.contains('Shane_Evs')]
# df = df[~df['text'].str.contains('Evs_Dubai')]

dfs_4 = [open_json_as_dataframe(f"../data/auto_trader_uk-26-Jan-2021.json")[0]]
dfs_5 = [open_json_as_dataframe(f"../data/beisgovuk-26-Jan-2021.json")[0]]
dfs_6 = [open_json_as_dataframe(f"../data/colin_mckerrache-26-Jan-2021.json")[0]]
dfs_7 = [open_json_as_dataframe(f"../data/smmt-26-Jan-2021.json")[0]]
dfs_8 = [open_json_as_dataframe(f"../data/tesla-26-Jan-2021.json")[0]]
dfs_9 = [open_json_as_dataframe(f"../data/toyota_uk-26-Jan-2021.json")[0]]
dfs_10 = [open_json_as_dataframe(f"../data/volkswagon-uk-26-Jan-2021.json")[0]]
df = pd.concat(dfs_4 + dfs_5 + dfs_6 + dfs_7 + dfs_8 + dfs_9 + dfs_10)
print(df.shape)
df = df[df['text'].str.contains('ev|EV|electric vehicle|electric car|Electric Vehicle|Electric Car')]
print(df.shape)

# *********************************

tweet_list = df.text.to_list()
tweet_list = clean(tweet_list)

tweet_polarity = [TextBlob(tweet).polarity for tweet in tweet_list]
tweet_subjectivity = [TextBlob(tweet).sentiment.subjectivity for tweet in tweet_list]
df["tb_polarity"] = tweet_polarity
df["subjectivity"] = tweet_subjectivity

weight_favourites = False
if weight_favourites:
    tot_favourites = (df["favorite_count"]+1).sum()
    df["favorite_weight"] = (df["favorite_count"] + 1)/tot_favourites
    df["weighted_polarity"] = df["favorite_weight"] * df["polarity"]

df.to_pickle("../processed_data/EV_promoters.pkl")  # Save as pickle
