"""Pulls recent tweets from a specified account."""
from datetime import date
import os

import json
import tweepy
from utils import open_json_as_dataframe, search_past_7_days, save_tweets_as_json

API_KEY = "0G6R5aK2a9Bvlq3EUfVgOsaEF"
API_KEY_SECRET = "QThazI5cEW2vj1p6yTbHguvyTIeaFQNVL8XEI6rxWDuTnnIaOp"
ACCESS_TOKEN = "1848494462-2bCP5LiDE8Lc8dKR09pxeqRJ0rIPO7seHWx5Ekk"
ACCESS_TOKEN_SECRET = "Q8omGl9o6s2qNYcqC5OLMf5IEBSXi5uAtV35i4iJvScwp"

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

filename = "test_filename"
userID = "@jack"

max_tweets = 20
# Handle pagination using the cursor object.
cursor = tweepy.Cursor(
    api.user_timeline,
    screen_name=userID,
    include_rts=True,
    exclude_replies=False,
    tweet_mode="extended"
).items(max_tweets)

# Gather the date, pausing 15 minutes any time the request limit is hit.
tweet_data = []
while True:
    try:
        tweet = cursor.next()
        tweet_data.append(tweet)
    except tweepy.TweepError:
        print("Entering except block, waiting...")
        time.sleep(60 * 15)
        print("Continuing search...")
        continue
    except StopIteration:
        # Entered when `max_tweets` reached.
        break

save_tweets_as_json(tweet_data, filename=f"../data/{filename}.json", search_term=userID, search_method="user_timeline")
df, v = open_json_as_dataframe(f"../data/{filename}-31-Jan-2021.json")
df.to_excel(f"{filename}.xlsx")
