import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

userID = "realDonaldTrump"

tweets = api.user_timeline(
    screen_name=userID,
    count=200,  # 200 is the maximum allowed count
    include_rts=False,
    tweet_mode="extended"
)
