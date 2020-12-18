"""This searches for a specified term."""
import tweepy
from utils import search_past_7_days

# Variables that contains the user credentials to access Twitter API
API_KEY = "XLr54F1FRNE7z5vuCgQDGhREq"
API_KEY_SECRET = "55jFS2OhBPgrdr343VjmpAfgz1uf3FzA6GQ1OqYeip93OGHsS0"
ACCESS_TOKEN = "1848494462-VngJMNvp0aYHZRsJTawjfAoLlkxhZWGS25IZNOU"
ACCESS_TOKEN_SECRET = "GONapY2u3aOALC35KRXOcm0y7uz4eODwAq8ZVRSAGsuFy"

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Scrape twitter. Note a free twitter developer account only supports up to
# 100 tweets per request. If a premium account is created the `search_full_archive`
# method can be used.
# See: https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/overview

search_term = "golf"
tweets = search_past_7_days(search_term, api, max_tweets=50)
print('hi')
for tweet in tweets:
    print(tweet.user.name)
    print(tweet.text)
    print("************************************")

print(f"Number of tweets scraped: {len(tweets)}")
