import matplotlib.pyplot as plt
import pandas as pd

tesla = pd.read_csv("tesla_stock_2017_2021.csv")

tesla["Date"] = pd.to_datetime(tesla['Date'])
tesla["Close/Last"] = tesla["Close/Last"].str.replace('$', '')
tesla["Close/Last"] = tesla["Close/Last"].astype(float)

x= tesla["Close/Last"][0]
print(type(x))

plt.plot(tesla["Date"], tesla["Close/Last"])
plt.show()