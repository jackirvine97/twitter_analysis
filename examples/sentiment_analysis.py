"""This plot runs a sentiment analysis on two inputs and plots results,
detailing polarity and subjectivity."""
import matplotlib.pyplot as plt
from textblob import TextBlob
import tweepy

# Authenticate to Twitter
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

pret_tweets = api.search(
    q="pret",
    lang="en",
    rpp=10,
    count=200
)
greggs_tweets = api.search(
    q="greggs",
    lang="en",
    rpp=10,
    count=200
)

pret_polarity = [TextBlob(tweet.text).sentiment.polarity for tweet in pret_tweets]
pret_subjectivity = [TextBlob(tweet.text).sentiment.subjectivity for tweet in pret_tweets]
greggs_polarity = [TextBlob(tweet.text).sentiment.polarity for tweet in greggs_tweets]
greggs_subjectivity = [TextBlob(tweet.text).sentiment.subjectivity for tweet in greggs_tweets]

# TODO: determine better way to visualise results.
plt.scatter(pret_polarity, pret_polarity, label="Pret")
plt.scatter(greggs_polarity, greggs_subjectivity, label="Greggs")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.xlim(-1.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.show()
