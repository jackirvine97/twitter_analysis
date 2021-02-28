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


def create_tweet_data_dict_from_tweet_obj(tweet, tweet_attrs):
    """Pulls relevant data from a `Tweet` object into a dictionary.

    tweet: tweepy.Tweet
        Tweepy `Tweet` object, returned from a search method.
    tweet_attrs: list
        List of attributes to be collected from `tweet`.

    """
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
    single_tweet_dict["search_method"] = "search_function"
    single_tweet_dict["is_rt"] = True if hasattr(tweet, "retweeted_status") else False

    return single_tweet_dict


def create_tweet_data_dict_from_status_obj(status, tweet_attrs):
    """Pulls relevant data from a `Status` object into a dictionary.

    tweet: tweepy.Status
        Tweepy `Status` object, returned from a search method.
    tweet_attrs: list
        List of data to be collected from `Status`.

    """
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

    # Additional key-values accessed accessed through additional hierarchy.
    single_tweet_dict["created_at"] = status._json["created_at"]
    single_tweet_dict["hashtags"] = [entity["text"] for entity in status._json["entities"]["hashtags"]]
    single_tweet_dict["mentions"] = [entity["screen_name"] for entity in status._json["entities"]["user_mentions"]]
    user_dictionary = status._json["user"]
    single_tweet_dict["user_followers_count"] = user_dictionary["followers_count"]
    single_tweet_dict["user_screen_name"] = user_dictionary["screen_name"]
    single_tweet_dict["user_user_location"] = user_dictionary["location"]
    single_tweet_dict["search_method"] = "user_timeline"
    single_tweet_dict["is_rt"] = is_rt

    return single_tweet_dict


def save_tweets_as_json(tweet_list, *, filename, search_term, search_method="search"):
    """Extracts data from tweets and saves as JSON file.

    Parameters
    ----------
    tweet_list: list
        List of `Tweet` objects when `search` function is used. List of `Status`
        objects when `user_timeline` function is used.
    filename: str
        Name of JSON file to be saved, including relative path from working
        directory to target destination. JSON file extension (.json) will be
        appended automatically if not included in this argument.
    search_term: str
        Query term used to extract the tweets in `tweet_list`.
    search_method: str
        Default is "search". Must be a valid type of tweepy search. Search and
        user timeline queries currently supported.

    Notes
    -----
    This function handles saves from two types of searches. A tweepy search
    returns a list of `Tweet` objects, whereas a tweepy user timeline search
    returns a list of `Status` objects. These objects have different attributes,
    warranting different functions to collect the same data.

    The `reply count` attribute is only available with premium accounts.

    """
    SEARCH_KEYWORD = "search"
    USER_TIMELINE_KEYWORD = "user_timeline"
    search_methods = (SEARCH_KEYWORD, USER_TIMELINE_KEYWORD)
    if search_method.lower() not in search_methods:
        q = f"search_method must take one of the following values : {search_methods}"
        raise ValueError(q)

    data_dict, metadata, tweets = {}, {}, []

    search_date_str = date.today().strftime("%d-%b-%Y")

    tweet_attrs = ["id", "retweet_count", "favorite_count",
                   "in_reply_to_status_id", "in_reply_to_screen_name",
                   "in_reply_to_user_id", "source", "lang",  "geo",
                   "coordinates"]

    if search_method == SEARCH_KEYWORD:
        for tweet in tweet_list:
            tweets.append(create_tweet_data_dict_from_tweet_obj(tweet, tweet_attrs))
        num_tweets = len(tweet_list)
    elif search_method == USER_TIMELINE_KEYWORD:
        num_tweets = 0
        for tweet in tweet_list:
            tweets.append(create_tweet_data_dict_from_status_obj(tweet, tweet_attrs))
            num_tweets += 1  # As length of iterator unknown.

    metadata["date_collected"] = search_date_str
    metadata["search_term"] = f"User timeline: @{search_term}" if search_method == SEARCH_KEYWORD else f"search: {search_term}"
    metadata["num_tweets"] = num_tweets

    data_dict["metadata"] = metadata
    data_dict["tweets"] = tweets

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
    regrex_pattern = re.compile(
        pattern="["
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
