import tweepy

# Authenticate to Twitter and instantiate API.
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

userID = "realDonaldTrump"

tweets = api.user_timeline(
    screen_name=userID,
    count=200,  # 200 is the maximum allowed count
    include_rts=False,
    tweet_mode="extended"
)

for info in tweets:
    print("ID: {}".format(info.id))
    print(info.created_at)
    print(info.full_text)
    print("\n")
