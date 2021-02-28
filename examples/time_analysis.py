"""Plots results against tweet time."""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_pickle("../processed_data/ICE_ban_November_2020_sent.pkl")

freq = "15min"
df["created_at"] = pd.to_datetime(df['created_at'])
df = df[["created_at", "polarity"]]
# df.groupby(pd.Grouper(key='created_at', freq=freq)).count().plot()
df.groupby(pd.Grouper(key='created_at', freq=freq)).count().plot()
plt.show()