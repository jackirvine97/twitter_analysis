"""Temporary collection of utility functions.
TODO - determine better utility strategy to avoid import errors."""
from datetime import date
import json
import os
import time
import tweepy


def search_past_7_days(search_term, api, *, max_tweets=100, language="en"):
    """Returns specified number of tweets within the last 7 days.

    Parameters
    ----------
    search_term: :obj:`str`
        Search query.
    max_tweets: :obj:`int`
        Maximum number of tweets to be scraped. Default is 100.
    search_term: :obj:`str`
        Language of tweets.

    Returns
    -------
    :obj:`list`
        List of :obj:`tweepy.tweet` objects.

    """

    # Handle pagination using the cursor object.
    cursor = tweepy.Cursor(
        api.search,
        q=search_term,
        language=language,
        include_entities=True
    ).items(max_tweets)

    # Gather the date, pausing 15 minutes any time the request limit is hit.
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
            # Entered when `max_tweets` reached.
            break

    return tweet_data


def save_tweets_as_json(tweet_list, *, filename, search_term):
    """Extracts data from tweets and saves as JSON file.

    Parameters
    ----------
    tweet_list: :obj:`list`
        List of :obj:`Tweet` to be saved.
    search_term: :obj:`str`
        Query term used to extract the tweets in `tweet_list`.
    filename: :obj:`str`
        Name of JSON file to be saved, including relative path from working
        directory to target destination. JSON file extension (.json) will be
        appended automatically if not included in this argument.

    """

    data_dict = {}
    metadata = {}
    tweets = []

    search_date_str = date.today().strftime("%d-%b-%Y")

    metadata["date_collected"] = search_date_str
    metadata["num_tweets"] = len(tweet_list)
    metadata["search_term"] = search_term
    data_dict["metadata"] = metadata

    for tweet in tweet_list:
        single_tweet_dict = {}
        single_tweet_dict['text'] = tweet.text
        single_tweet_dict['created_at'] = tweet.created_at.strftime("%d-%b-%Y %H:%M:%S")
        single_tweet_dict['id_str'] = tweet.id_str
        single_tweet_dict['retweet_count'] = tweet.retweet_count
        single_tweet_dict['favorite_count'] = tweet.favorite_count
        single_tweet_dict['in_reply_to_screen_name'] = tweet.in_reply_to_screen_name
        user_dictionery = tweet._json['user']
        single_tweet_dict['followers_count'] = user_dictionery['followers_count']
        single_tweet_dict['screen_name'] = user_dictionery['screen_name']
        tweets.append(single_tweet_dict)
    data_dict["tweets"] = tweets

    root, ext = os.path.splitext(filename)
    if ext != ".json":
        print(f"The extension {ext} is invalid. Replacing with '.json'")
        ext = ".json"
    filename = f"{root}-{search_date_str}{ext}"

    with open(filename, 'w') as json_file:
        json.dump(data_dict, json_file)

    return
