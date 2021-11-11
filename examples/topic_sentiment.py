"""Plots topic volume, segmented by sentiment."""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import SemiMonthBegin

df_raw = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics_4.pkl")
df = df_raw.loc[df_raw["tb_polarity"] == 1.000]
df_1 = df_raw.loc[df_raw["tb_polarity"] == -1.000]
df = pd.concat([df]+[df_1])

rolling_period = 4

df = df_raw.loc[df_raw["lang"] == "en"]  # Filter out languages.
pos_df = df[["sentiment_status", "dominant_topic"]]
pos_df = pos_df.loc[pos_df["sentiment_status"] == "positive"]
pos_df = pos_df.groupby(by=["dominant_topic"]).count()
pos_df["topics"] = ["Sales", "Battery Technology", "Range", "Cost", "Charging Infrastructure"]

df = df_raw.loc[df_raw["lang"] == "en"]  # Filter out languages.
neg_df = df[["sentiment_status", "dominant_topic"]]
neg_df = neg_df.loc[neg_df["sentiment_status"] == "neutral"]
neg_df = neg_df.groupby(by=["dominant_topic"]).count()

df = df_raw.loc[df_raw["lang"] == "en"]  # Filter out languages.
neut_df = df[["sentiment_status", "dominant_topic"]]
neut_df = neut_df.loc[neut_df["sentiment_status"] == "negative"]
neut_df = neut_df.groupby(by=["dominant_topic"]).count()

fig, ax = plt.subplots()
bar1 = ax.barh(pos_df["topics"], pos_df["sentiment_status"], color="#57ae00", label="Positive", height=0.54)
bar2 = ax.barh(pos_df["topics"], neut_df["sentiment_status"], color="#ffbc00", label="Neutral", left=pos_df.sentiment_status, height=0.54)
left_2 = [x+y for x, y in zip(pos_df["sentiment_status"], neut_df["sentiment_status"])]

bar3 = ax.barh(pos_df["topics"], neg_df["sentiment_status"], color="#b50000", label="Negative", left=left_2, height=0.54)
ax.set_xlabel("Number of Tweets", size=10)
ax.set_ylabel("hh")
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend(loc='lower right')

plt.title("Key Topics Discussed")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
print(pos_df)

print(neg_df)

print(neut_df)

for index in range(5):
	tot = pos_df.sentiment_status[index] +neut_df.sentiment_status[index] +  neg_df.sentiment_status[index]
	print(f'pos %{pos_df.sentiment_status[index]*100/tot}')
	print(f'neut %{neut_df.sentiment_status[index]*100/tot}')
	print(f'neg %{neg_df.sentiment_status[index]*100/tot}')
	print('********** ')