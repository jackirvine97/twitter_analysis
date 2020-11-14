"""This searches for trending topics in a specified location."""
import tweepy

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# id – The Yahoo! Where On Earth ID of the location to return trending
loc_id = 44418  # London
trends_result = api.trends_place(loc_id)
for trend in trends_result[0]["trends"]:
    print(trend["name"])