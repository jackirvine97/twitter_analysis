"""Plots results against tweet time."""
import datetime

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import SemiMonthBegin

df_raw = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_2.pkl")

rolling_period = 4

df = df_raw.loc[df_raw["lang"] == "en"]  # Filter out languages.
df["created_at"] = pd.to_datetime(df['created_at'])
df = df[["created_at", "tb_polarity", "favorite_count", "retweet_count"]]
df["favorite_count"] = df["favorite_count"] + 1
df.set_index("created_at", inplace=True)


def weighted_average(arr):
    """Returns aggregated average, weighted by favourite count."""
    fav_sum = (arr["favorite_count"] + 0.005*arr["retweet_count"]).sum()
    weighted_average = (arr["tb_polarity"] * (arr["favorite_count"] + 0.005*arr["retweet_count"])/fav_sum).sum()
    return weighted_average


df = df.groupby(pd.Grouper(freq=SemiMonthBegin(day_of_month=16))).apply(weighted_average)
df.index = df.index + to_offset("14d")
df.rename(index={0: "polarity"}, inplace=True)


fig, ax = plt.subplots()
lns1 = ax.plot(df.rolling(rolling_period).mean().index, df.rolling(rolling_period).mean(),
        color="slategrey", label="Sentiment Score")
ax.set_xlabel("Date")
ax.set_ylim(0.085, 0.19)
ax.set_xlim([datetime.date(2020, 3, 1), datetime.date(2021, 3, 20)])
ax.set_ylabel("Engagement Weighted Average Sentiment")

tesla = pd.read_csv("tesla_stock_2017_2021.csv")
tesla["Date"] = pd.to_datetime(tesla['Date'])
tesla["Close/Last"] = tesla["Close/Last"].str.replace('$', '')
tesla["Close/Last"] = tesla["Close/Last"].astype(float)

ax2 = ax.twinx()
lns2 = ax2.plt(tesla["Date"], tesla["Close/Last"], label="Tesla Stock Price", color="plum")
ax2.set_ylabel("Value ($)")
ax2.set_ylim(-5, 910)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Create custom legend by unpacking tuples containing handles and using 
# only one set of unpacked labels along with set of unpacked empty strings
# (using None instead of empty strings gives the same result)
ax.legend((*handles1, *handles2), (*len(labels1)*[''], *labels2),
             loc='upper left', ncol=2, handlelength=3, edgecolor='black',
             borderpad=0.7, handletextpad=1.5, columnspacing=0)

plt.show()






