import datetime

import pandas as pd
import json
import matplotlib.pyplot as plt

from utils import open_json


day_list = open_json("Counts_Nov20_Mar_21.json")["results"]
for index in range(2, 5):
    day_list += open_json(f"Counts_Nov20_Mar_21_{index}.json")["results"]
for index in range(1, 12):
    day_list += open_json(f"Counts_Jan20_Nov20_{index}.json")["results"]

times = [datetime.datetime.strptime(day["timePeriod"], '%Y%m%d%H%M') for day in day_list]
tweet_count = [int(day["count"]) for day in day_list]

df = pd.DataFrame(times)
df["Count"] = tweet_count
df.columns = ["date", "count"]
df = df.sort_values("date").reset_index(drop=True)

plt.plot(df["date"], df["count"], label="Actual", color="deepskyblue")
df = df.set_index("date")
plt.plot(df.rolling(14).mean().index, df.rolling(14).mean(), label="14 Day Rolling Average", linestyle="dashed", color="slategrey", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.legend(loc=4)
plt.show()
