"""This searches for a specified term."""
import tweepy
from utils import (save_tweets_as_json, search_past_7_days)

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Scrape twitter. Note a free twitter developer account only supports up to
# 100 tweets per request. If a premium account is created the `search_full_archive`
# method can be used.
# See: https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/overview

search_term = "golf"
tweets = search_past_7_days(search_term, api, max_tweets=100)
save_tweets_as_json(tweets, filename='../data/golf.xls', search_term=search_term)

for tweet in tweets:
    print(tweet.user.name)
    print(tweet.text)
    print("************************************")

print(f"Number of tweets scraped: {len(tweets)}")
