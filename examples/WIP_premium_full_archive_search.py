"""
Plan - Do 18th. try 2500. If we pull less in, then judge how many days to go into the future. If less then try and cover the whole day.

ICE_ban_November_2020_2: 19th0000 - 18th000 2500 asked
"""
from datetime import date
import os
import re
import time
import json
import pandas as pd

import tweepy

from utils import open_json_as_dataframe
from config import (API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

print("Is the search term correct?")
input()
print("Have you changed the save name to avoid overwrite?")
input()
print("change excel name?")
input()
print("Dates correct?")
input()
print("How many tweets?")
input()
print("Max results 100 or 500")
input()
"""
**********************
Query Terms
**********************
"""

# Copy paste dates in
# Check dates and compress
# Check name
# Excel name

toDate   = "202003122100"  # Most recent date.
fromDate = "202003010000"  # Oldest date.
filename = "EV-Mar-12_2020_4500"
search_term = "(electric vehicle OR electric car)"
environment_name = "JackDemoOrig"

excel = "EV-Mar-12_2020_4500.xlsx"


"""
**********************
Extract Tweets
**********************
"""

# Build Cursor.
max_tweets = 4500
cursor = tweepy.Cursor(
    api.search_full_archive,
    environment_name=environment_name,
    query=search_term,
    toDate=toDate,
    fromDate=fromDate,
    maxResults=100
).items(max_tweets)


# Gather the date, pausing 15 minutes any time the request limit is hit.
tweet_data = []
while True:
    try:
        tweet = cursor.next()
        tweet_data.append(tweet)
    except tweepy.TweepError as e:
        print(e)
        print("Entering except block, waiting...")
        time.sleep(60 * 15)
        print("Continuing search...")
        continue
    except StopIteration:
        # Entered when `max_tweets` reached.
        break

"""
**********************
Save to JSON - temp - must be added to utils.
**********************
"""

data_dict, metadata, tweets = {}, {}, []

search_date_str = date.today().strftime("%d-%b-%Y")

tweet_attrs = ["id", "retweet_count", "favorite_count",
               "in_reply_to_status_id", "in_reply_to_screen_name",
               "in_reply_to_user_id", "source", "lang",  "geo",
               "coordinates"]

num_tweets = 0
for status in tweet_data:
    single_tweet_dict = {}
    num_tweets += 1

    if status.truncated:
        original_text = status.extended_tweet['full_text']
    else:
        original_text = status.text

    # Ensure text for RTs is fully captured.
    try:
        key = 'extended_tweet'
        rt_text = status.retweeted_status._json[key]["full_text"]
        user_screen_name = status.retweeted_status.user._json['screen_name']
        single_tweet_dict["text"] = f"RT @{user_screen_name}{rt_text}" if original_text.startswith("RT @") else original_text
        is_rt = True

    except (KeyError, AttributeError):
        single_tweet_dict["text"] = original_text
        is_rt = False

    for attr in tweet_attrs:
        single_tweet_dict[attr] = getattr(status, attr)

    # Additional attrs accessed accessed through additional hierarchy.
    single_tweet_dict["created_at"] = status.created_at.strftime("%d-%b-%Y %H:%M:%S")
    single_tweet_dict["hashtags"] = [entity["text"] for entity in status.entities["hashtags"]]
    single_tweet_dict["mentions"] = [entity["screen_name"] for entity in status.entities["user_mentions"]]
    user_dictionary = status._json["user"]
    single_tweet_dict["user_followers_count"] = user_dictionary["followers_count"]
    single_tweet_dict["user_screen_name"] = user_dictionary["screen_name"]
    single_tweet_dict["user_user_location"] = user_dictionary["location"]
    single_tweet_dict["search_method"] = "search_full_archive"
    single_tweet_dict["rt"] = is_rt

    tweets.append(single_tweet_dict)


metadata["date_collected"] = search_date_str
metadata["search_term"] = search_term
metadata["num_tweets"] = num_tweets

data_dict["metadata"] = metadata
data_dict["tweets"] = tweets


root, ext = os.path.splitext(f"../data/{filename}")
if ext != ".json":
    print(f"The extension {ext} is invalid. Replacing with '.json'")
    ext = ".json"
filename = f"{root}-{search_date_str}{ext}"

with open(filename, "w") as json_file:
    json.dump(data_dict, json_file)

df, meta = open_json_as_dataframe(f"{filename}")
df.to_excel(excel)
