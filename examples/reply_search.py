"""This script queries all replies of a specified tweet.
This script is WIP and not to merge into master."""
import tweepy

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

search_term = "golf"
tweets = []

for tweet in api.search("Test30Day", search_term=search_term, max_results=100):
    tweets.append(tweet)
    print(tweet.user.name)
    print(tweet.text)
    print(tweet.date)
    print("************************************")

print(f"Number of tweets scraped: {len(tweets)}")