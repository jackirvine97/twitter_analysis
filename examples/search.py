"""This searches for a specified term.

Note a free twitter developer account only supports up to
100 tweets per request. If a premium account is created the `search_full_archive`
method can be accessed.
See: https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/overview.

"""
import tweepy
from utils import (open_json_as_dataframe, save_tweets_as_json, search_past_7_days)

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

search_term = "physics AND maths"
tweets = search_past_7_days(search_term, api, max_tweets=10)

save_tweets_as_json(tweets, filename="../data/subjects.json", search_term=search_term)
print(f"Number of tweets pulled: {len(tweets)}")
