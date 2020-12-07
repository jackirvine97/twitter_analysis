"""This plot runs a sentiment analysis on two inputs and plots results,
detailing polarity and subjectivity."""
import matplotlib.pyplot as plt
from textblob import TextBlob
import tweepy

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

happy_tweets = api.search(
    q="happy",
    lang="en",
    rpp=10,
    count=200
)
brexit_tweets = api.search(
    q="brexit",
    lang="en",
    rpp=10,
    count=200
)

happy_polarity = [TextBlob(tweet.text).sentiment.polarity for tweet in happy_tweets]
happy_subjectivity = [TextBlob(tweet.text).sentiment.subjectivity for tweet in happy_tweets]
brexit_polarity = [TextBlob(tweet.text).sentiment.polarity for tweet in brexit_tweets]
brexit_subjectivity = [TextBlob(tweet.text).sentiment.subjectivity for tweet in brexit_tweets]

neutral_threshold = 0.05
happy_number_positive = sum(pol > neutral_threshold for pol in happy_polarity)
happy_number_negative = sum(pol < -neutral_threshold for pol in happy_polarity)
happy_number_neutral = len(happy_polarity) - happy_number_positive - happy_number_negative

brexit_number_positive = sum(pol > neutral_threshold for pol in brexit_polarity)
brexit_number_negative = sum(pol < neutral_threshold for pol in brexit_polarity)
brexit_number_neutral = len(brexit_polarity) - brexit_number_positive - brexit_number_negative

# TODO: determine better way to visualise results.
plt.scatter(happy_polarity, happy_subjectivity, label="Happy")
plt.scatter(brexit_polarity, brexit_subjectivity, label="Brexit")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.xlim(-1.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.show()

labels = ['Positive', 'Neutral', 'Negative']
sizes = [happy_number_positive, happy_number_neutral, happy_number_negative]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        startangle=90)
ax1.set_title("Happy")
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

labels = ['Positive', 'Neutral', 'Negative']
sizes = [brexit_number_positive, brexit_number_neutral, brexit_number_negative]
fig2, ax2 = plt.subplots()
ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
        startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax2.set_title("Brexit")

plt.show()
