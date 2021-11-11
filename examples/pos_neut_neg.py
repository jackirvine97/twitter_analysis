import pandas as pd
import numpy as np

df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics_4.pkl")
print(df.shape)

conditions = [
    (df['tb_polarity'] < 0),
    (df['tb_polarity'] == 0),
    (df['tb_polarity'] > 0)]
choices = ['negative', 'neutral', 'positive']
df['sentiment_status'] = np.select(conditions, choices)


df.to_pickle("../processed_data/EVs_2020_Tweets_sent_topics_4.pkl")
