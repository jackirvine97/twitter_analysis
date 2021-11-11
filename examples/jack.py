"""Plots results against tweet time."""
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import SemiMonthBegin

rolling_period=4

df_raw = pd.read_pickle("../processed_data/ICE_Ban_November_2020_sent.pkl")

df_raw["created_at"] = pd.to_datetime(df_raw['created_at'])
df = df_raw[["created_at", "polarity", "favorite_count", "retweet_count"]]
df["favorite_count"] = df["favorite_count"] + 1
df.set_index("created_at", inplace=True)


def weighted_average(arr):
    """Returns aggregated average, weighted by favourite count."""
    fav_sum = (arr["favorite_count"] + 0.005*arr["retweet_count"]).sum()
    weighted_average = (arr["polarity"] * (arr["favorite_count"] + 0.005*arr["retweet_count"])/fav_sum).sum()
    return weighted_average


df = df.groupby(pd.Grouper(freq=SemiMonthBegin(day_of_month=16))).apply(weighted_average)
df.index = df.index + to_offset("14d")

print(df)
df.rename(index={0: "polarity"}, inplace=True)

# plt.plot(df.index, df, alpha=0.4, color="deepskyblue", label="Actual")

plt.plot(df.rolling(rolling_period).mean().index, df.rolling(rolling_period).mean(),
         linewidth=3, color="slategrey", label="Combined Dataset")
plt.title("Engagement Weighted Average Sentiment")


plot_topics = False
if plot_topics:
    topics = [0, 1, 2, 3, 4]
    topic_labels = ["Battery Technology", "EV Sales", "Range", "Cost", "Charging Infrastructure"]
    colours= ["seagreen", "orchid", "lightcoral", "lightsteelblue", "sienna"]
    for topic, label, colour in zip(topics, topic_labels, colours):
        df = df_raw.loc[df_raw["dominant_topic"] == topic]
        df = df.loc[df["lang"] == "en"]  # Filter out languages.
        df["created_at"] = pd.to_datetime(df['created_at'])
        df = df[["created_at", "tb_polarity", "favorite_count", "retweet_count"]]
        df["favorite_count"] = df["favorite_count"] + 1
        df.set_index("created_at", inplace=True)
        df = df.groupby(pd.Grouper(freq=SemiMonthBegin(day_of_month=16))).apply(weighted_average)
        df.index = df.index + to_offset("14d")
        df.rename(index={0: "polarity"}, inplace=True)
        plt.plot(df.rolling(6).mean().index, df.rolling(6).mean(), label=label, alpha=1, color=colour, linestyle="dotted")

plt.legend()
plt.xlabel("Date")
plt.ylabel("Engagement Weighted 2 Month Average Sentiment")

plt.show()




# df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent.pkl")
# df = df.loc[df["lang"] == "en"]  # Filter out languages.
# print(df.shape)
# df["created_at"] = pd.to_datetime(df['created_at'])
# df = df[["created_at", "weighted_polarity"]]
# df.set_index("created_at", inplace=True)

# df2 = df.resample(SemiMonthBegin(day_of_month=16)).sum()
# df2.index = df2.index + to_offset("14d")
# df2["weighted_polarity"] = df2["weighted_polarity"] * 29
# df2.plot()
# df2.weighted_polarity.rolling(4).mean().plot()

# plt.title("group weighted mean")
# plt.show()



# df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent.pkl")
# df = df.loc[df["lang"] == "en"]  # Filter out languages.
# print(df.shape)
# df["created_at"] = pd.to_datetime(df['created_at'])
# df = df[["created_at", "polarity"]]
# df.set_index("created_at", inplace=True)

# df2 = df.resample(SemiMonthBegin(day_of_month=16)).mean()
# df2.index = df2.index + to_offset("14d")
# df2["polarity"] = df2["polarity"]
# df2.plot()
# df2.polarity.rolling(4).mean().plot()

# plt.title("mean")
# plt.show()
