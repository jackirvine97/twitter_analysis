import matplotlib.pyplot as plt
import tweepy
from wordcloud import WordCloud

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

search_term = "#pret"
number_of_tweets = 600

tweets = tweepy.Cursor(api.search, q=search_term, lang="en").items(number_of_tweets)

cloud = ""
tweet_count = 0
for each in tweets:
    cloud = cloud + each.text
    tweet_count += 1

# TODO: determine better technique for stopwords.
stopwords = ("http", "https", "t", "co", "in", "the", "to", "it", "if")

wordcloud = WordCloud(background_color="white", width=1600, height=800, stopwords=stopwords).generate(cloud)

print(f"Number of tweets processed: {tweet_count}")

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
