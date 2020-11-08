"""This searches for a specified term in a tweet."""
import tweepy

# Authenticate to Twitter
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Create API object
api = tweepy.API(auth)

# Scrape twitter. Note a free twitter developer account only supports up to
# 100 tweets per request. If a premium account is created the `search_full_archive`
# method can be used.
search_term = "golf"
tweets = []
for tweet in api.search(q=search_term, lang="en", rpp=10, count=100):
    tweets.append(tweet)
    print(tweet.user.name)
    print(tweet.text)
    print("************************************")

print(f"Number of tweets scraped: {len(tweets)}")