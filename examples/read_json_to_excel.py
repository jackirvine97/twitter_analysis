"""Reads saved tweet JSON to dataframe."""
import pandas as pd
from utils import open_json_as_dataframe

output_filename = "../excel_placeholder/ICE_Ban_Tweets.xlsx"

# dfs_1 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-31-Jan-2021.json")[0] for index in range(1, 5)]
# dfs_2 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-01-Feb-2021.json")[0] for index in range(5, 10)]
# df = pd.concat(dfs_1 + dfs_2)
df = open_json_as_dataframe(f"../data/EV-Mar-2021_1-21-Mar-2021.json")[0]
df.to_excel(output_filename)
