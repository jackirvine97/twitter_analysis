"""Temporary collection of utility functions.
TODO:
- determine better utility strategy to avoid import errors.
- keywords should be extracted as constants.
"""
from datetime import date
import os
import re
import time

import json
import pandas as pd
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
        include_entities=True,
        tweet_mode="extended"
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

    Notes
    -----
    The `reply count` attribute is only available with premium accounts.
    """

    data_dict = {}
    metadata = {}
    tweets = []

    search_date_str = date.today().strftime("%d-%b-%Y")

    metadata["date_collected"] = search_date_str
    metadata["num_tweets"] = len(tweet_list)
    metadata["search_term"] = search_term
    data_dict["metadata"] = metadata

    tweet_attrs = ["id", "retweet_count", "favorite_count",
                   "in_reply_to_status_id", "in_reply_to_screen_name",
                   "in_reply_to_user_id", "source", "lang",  "geo",
                   "coordinates"]

    for tweet in tweet_list:
        single_tweet_dict = {}
        try:
            single_tweet_dict["text"] = f"RT @{tweet.retweeted_status.user._json['screen_name']}{tweet.retweeted_status.full_text}" if tweet.full_text.startswith("RT @") else tweet.full_text
        except AttributeError:
            single_tweet_dict["text"] = tweet.full_text
        for attr in tweet_attrs:
            single_tweet_dict[attr] = getattr(tweet, attr)
        # Additional attrs accessed accessed through additional hierarchy.
        single_tweet_dict["created_at"] = tweet.created_at.strftime("%d-%b-%Y %H:%M:%S")
        single_tweet_dict["hashtags"] = [entity["text"] for entity in tweet.entities["hashtags"]]
        single_tweet_dict["mentions"] = [entity["screen_name"] for entity in tweet.entities["user_mentions"]]
        user_dictionary = tweet._json["user"]
        single_tweet_dict["user_followers_count"] = user_dictionary["followers_count"]
        single_tweet_dict["user_screen_name"] = user_dictionary["screen_name"]
        single_tweet_dict["user_user_location"] = user_dictionary["location"]
        tweets.append(single_tweet_dict)
    data_dict["tweets"] = tweets

    root, ext = os.path.splitext(filename)
    if ext != ".json":
        print(f"The extension {ext} is invalid. Replacing with '.json'")
        ext = ".json"
    filename = f"{root}-{search_date_str}{ext}"

    with open(filename, "w") as json_file:
        json.dump(data_dict, json_file)

    return


def save_from_user_timeline_search(tweet_list, *, filename, user_id):
    data_dict = {}
    metadata = {}
    tweets = []

    search_date_str = date.today().strftime("%d-%b-%Y")

    metadata["date_collected"] = search_date_str
    metadata["search_term"] = f"User timeline: @{user_id}"
    data_dict["metadata"] = metadata

    tweet_attrs = ["id", "retweet_count", "favorite_count",
                   "in_reply_to_status_id", "in_reply_to_screen_name",
                   "in_reply_to_user_id", "source", "lang",  "geo",
                   "coordinates"]

    num_tweets = 0
    for status in tweet_list:
        num_tweets += 1
        single_tweet_dict = {}
        try:
            key = 'retweeted_status'
            original_text = status._json["full_text"]
            rt_text = status._json[key]["full_text"]
            user_screen_name = status._json[key]['user']['screen_name']
            single_tweet_dict["text"] = f"RT @{user_screen_name}{rt_text}" if original_text.startswith("RT @") else original_text
            is_rt = True
        except KeyError:
            single_tweet_dict["text"] = status._json["full_text"]
            is_rt = False

        for attr in tweet_attrs:
            single_tweet_dict[attr] = status._json[attr]
        # Additional attrs accessed accessed through additional hierarchy.
        single_tweet_dict["created_at"] = status._json["created_at"]
        single_tweet_dict["hashtags"] = [entity["text"] for entity in status._json["entities"]["hashtags"]]
        single_tweet_dict["mentions"] = [entity["screen_name"] for entity in status._json["entities"]["user_mentions"]]
        user_dictionary = status._json["user"]
        single_tweet_dict["user_followers_count"] = user_dictionary["followers_count"]
        single_tweet_dict["user_screen_name"] = user_dictionary["screen_name"]
        single_tweet_dict["user_user_location"] = user_dictionary["location"]
        single_tweet_dict["search_method"] = "user_timeline"
        single_tweet_dict["is_rt"] = is_rt
        tweets.append(single_tweet_dict)
    data_dict["tweets"] = tweets
    data_dict["metadata"]["num_tweets"] = num_tweets

    root, ext = os.path.splitext(f"../data/{filename}")
    if ext != ".json":
        print(f"The extension {ext} is invalid. Replacing with '.json'")
        ext = ".json"
    filename = f"{root}-{search_date_str}{ext}"

    with open(filename, "w") as json_file:
        json.dump(data_dict, json_file)

    return


def open_json(filename):
    """Opens JSON file a dictionary.

    Parameters
    ----------
    filename :obj:`str`
        Name of JSON file being loaded.

    Returns
    -------
    :obj:`dict`
        Dictionary containing JSON data.

    """
    with open(filename) as json_file:
        data_dict = json.load(json_file)
    return data_dict


def open_json_as_dataframe(filename):
    """Converts JSON data to pandas dataframe.

    Parameters
    ----------
    filename : str
        Name of JSON file being loaded.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with information on tweets.
    dict
        Dictionary containing query metadata.

    """
    data_dict = open_json(filename)
    metadata_dict = data_dict["metadata"]
    df = pd.DataFrame(data_dict["tweets"])
    df.index.name = "tweet_id"
    return df, metadata_dict


def de_emojify(text):
    """Removes emojis from a string.

    Parameters
    ----------
    text : str
        String to remove emojis from.

    Returns
    -------
    text : str
        String with any emojis removed.

    Notes
    -----
    See https://stackoverflow.com/questions/33404752/
        removing-emojis-from-a-string-in-python

    """
    regrex_pattern = re.compile(pattern="["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def clean(docs):
    """Removes emails, new lines, single quotation marks and urls, emojis.

    Parameters
    ----------
    doc_list
        List of documents to be cleaned.
    Returns
    -------
    list
        List of cleaned documents.

    """
    docs_clean = [re.sub('\S*@\S*\s?', '', sent) for sent in docs]
    docs_clean = [re.sub('\s+', ' ', sent) for sent in docs_clean]
    docs_clean = [re.sub("\'", "", sent) for sent in docs_clean]
    docs_clean = [re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", sent) for sent in docs_clean]
    docs_clean = [de_emojify(sent) for sent in docs_clean]
    return docs_clean
