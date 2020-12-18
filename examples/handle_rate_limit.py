"""This script uses exception handling to increase the number of real time
tweets that can be scraped."""
import time
import tweepy

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

search_term = "golf"
num_tweets = 1

# Handle pagination using the cursor object.
cursor = tweepy.Cursor(
    api.search,
    q=search_term,
    include_entities=True).items(num_tweets)

# Gather the date, pausing 15 minutes anytime the request limit is hit.
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
        # Entered when `num_tweets` reached.
        break

print(len(tweet_data))
