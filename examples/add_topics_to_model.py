"""Joins topic lookup to model."""
import pandas as pd

df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics.pkl")

topic_dict = {
    "index": [0, 1, 2, 3, 4],
    "topic_name": ["EV Sales", "Cost", "Battery Technology", "Charging Infrastructure", "Range"]
}
df_lookup = pd.DataFrame(topic_dict)
df_lookup.set_index("index", inplace=True)

df = df.join(df_lookup, on="dominant_topic")
df.to_pickle("../processed_data/EVs_2020_Tweets_sent_topics.pkl")
