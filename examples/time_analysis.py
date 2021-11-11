"""Plots results against tweet time."""
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import SemiMonthBegin


rolling_period=4
df_raw = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics_4.pkl")
# df = pd.read_pickle("../processed_data/EV_promoters.pkl")
# "../processed_data/EV_promoters.pkl"

df_raw["created_at"] = pd.to_datetime(df_raw['created_at'])
df = df_raw[["created_at", "tb_polarity", "favorite_count", "retweet_count"]]
# df["favorite_count"] = df["favorite_count"] + 1
df.set_index("created_at", inplace=True)


def weighted_average(arr):
    """Returns aggregated average, weighted by favourite count."""
    fav_sum = (arr["favorite_count"] + 0.005*arr["retweet_count"]).sum()
    weighted_average = (arr["tb_polarity"] * (arr["favorite_count"] + 0.005*arr["retweet_count"])/fav_sum).sum()
    return weighted_average

# df = df.groupby(pd.Grouper(freq='Q')).apply(weighted_average)
df = df.groupby(pd.Grouper(freq=SemiMonthBegin(day_of_month=16))).apply(weighted_average)
df.index = df.index + to_offset("14d")
df.rename(index={0: "polarity"}, inplace=True)

# plt.plot(df.index, df, alpha=0.4, color="deepskyblue", label="Actual")
plot_combined = False
if plot_combined:
plt.plot(df.rolling(rolling_period).mean().index, df.rolling(rolling_period).mean(),
         linewidth=3, color="slategrey", label="Combined Dataset")


plot_topics = True
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


plt.xlabel("Date")
plt.ylim(0.075, 0.25)
plt.title("Trend in Sentiment Score")
plt.ylabel("Engagement Weighted Average Sentiment Score")
plt.legend(loc='upper left')





# bar charts for topics


# df_raw = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics_4.pkl")
# df = df_raw.loc[df_raw["tb_polarity"] == 1.000]
# df_1 = df_raw.loc[df_raw["tb_polarity"] == -1.000]
# df = pd.concat([df]+[df_1])

# rolling_period = 4

# df = df_raw.loc[df_raw["lang"] == "en"]  # Filter out languages.
# pos_df = df[["sentiment_status", "dominant_topic"]]
# pos_df = pos_df.loc[pos_df["sentiment_status"] == "positive"]
# pos_df = pos_df.groupby(by=["dominant_topic"]).count()
# pos_df["topics"] = ["Sales", "Battery Technology", "Range", "Cost", "Charging Infrastructure"]

# df = df_raw.loc[df_raw["lang"] == "en"]  # Filter out languages.
# neg_df = df[["sentiment_status", "dominant_topic"]]
# neg_df = neg_df.loc[neg_df["sentiment_status"] == "neutral"]
# neg_df = neg_df.groupby(by=["dominant_topic"]).count()

# df = df_raw.loc[df_raw["lang"] == "en"]  # Filter out languages.
# neut_df = df[["sentiment_status", "dominant_topic"]]
# neut_df = neut_df.loc[neut_df["sentiment_status"] == "negative"]
# neut_df = neut_df.groupby(by=["dominant_topic"]).count()

# fig, ax = plt.subplots()
# bar1 = ax.barh(pos_df["topics"], pos_df["sentiment_status"], color="#57ae00", label="Positive", height=0.53)
# bar2 = ax.barh(pos_df["topics"], neut_df["sentiment_status"], color="#ffbc00", label="Neutral", left=neg_df.sentiment_status, height=0.53)
# left_2 = [x+y for x, y in zip(neg_df["sentiment_status"], neut_df["sentiment_status"])]

# bar3 = ax.barh(pos_df["topics"], neg_df["sentiment_status"], color="#b50000", label="Negative", left=left_2, height=0.53)
# ax.set_xlabel("Number of Tweets")
# ax.get_xaxis().set_major_formatter(
#     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# plt.legend()
# plt.title("Key Topics Discussed")
# plt.tight_layout()
# plt.gca().invert_yaxis()





# df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics_4.pkl")

# df["created_at"] = pd.to_datetime(df['created_at'])
# df = df[["created_at", "tb_polarity", "favorite_count", "retweet_count"]]
# df["favorite_count"] = df["favorite_count"] + 1
# df.set_index("created_at", inplace=True)


# def weighted_average(arr):
#     """Returns aggregated average, weighted by favourite count."""
#     fav_sum = (arr["favorite_count"] + 0.005*arr["retweet_count"]).sum()
#     weighted_average = (arr["tb_polarity"] * (arr["favorite_count"] + 0.005*arr["retweet_count"])/fav_sum).sum()
#     return weighted_average

# #df = df.groupby(pd.Grouper(freq='Q')).apply(weighted_average)
# df = df.groupby(pd.Grouper(freq=SemiMonthBegin(day_of_month=16))).apply(weighted_average)
# df.index = df.index + to_offset("14d")
# df.rename(index={0: "polarity"}, inplace=True)

# # plt.plot(df.index, df, alpha=0.4, color="deepskyblue", label="Actual")

# plt.plot(df.rolling(rolling_period).mean().index, df.rolling(rolling_period).mean(),
#          linewidth=2, color="slategrey", label="Full Dataset")
# plt.title("Engagement Weighted Average Sentiment")
# plt.legend()
# plt.show()






# df = pd.read_pickle("../processed_data/ICE_Ban_November_2020_sent.pkl")

# print(df.polarity.mean())

# favourites = df.favorite_count.to_list()
# retweets = df.retweet_count.to_list()
# tweet_list = df.text.to_list()
# tweet_polarity = df.polarity.to_list()
# tweet_subjectivity = df.subjectivity.to_list()

# neutral_threshold = 0
# strong_threshold = 0.5
# tweet_num_strong_positive = sum(pol > strong_threshold for pol in tweet_polarity)
# tweet_num_strong_negative = sum(pol < -strong_threshold for pol in tweet_polarity)
# tweet_num_weak_positive = sum((pol > neutral_threshold) and (pol <= strong_threshold) for pol in tweet_polarity)
# tweet_num_weak_negative = sum((pol < -neutral_threshold) and (pol >= -strong_threshold) for pol in tweet_polarity)
# tweet_num_neutral = (len(tweet_polarity) - tweet_num_strong_positive
#                      - tweet_num_strong_negative - tweet_num_weak_positive
#                      - tweet_num_weak_negative)
# plt.figure()
# # Plot scatter map.
# cm = plt.cm.get_cmap('GnBu')
# sc = plt.scatter(tweet_polarity, tweet_subjectivity, s=favourites, alpha=0.65,
#                  c=retweets, cmap=cm, vmin=-18, vmax=50)
# plt.xlabel("Sentiment Score (How Positive?)")
# plt.ylabel("Subjectivity (How Emotive?)")
# plt.title('Tweet Sentiment vs. Subjectivity vs. Engagement')
# plt.xlim(-1.05, 1.05)
# plt.ylim(-0.05, 1.05)











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
# 
plt.show()
