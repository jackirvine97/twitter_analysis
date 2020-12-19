"""Temporary collection of utility functions.
TODO - determine better utility strategy to avoid import errors."""
from datetime import date
import json
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
        include_entities=True).items(max_tweets)

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


def save_tweets_as_json(tweet_list, search_term, filename):

    data_dict = {}
    metadata = {}
    tweet_dict = {}

    metadata["date_collected"] = date.today()
    metadata["num_tweets"] = len(tweet_list)
    metadata["search_term"] = search_term
    metadata[""] = 
    metadata[""] = 

    data_dict["metadata"] = metadata

    # Save `data_dict`as json file.
    json_file = open(filename, 'wb')
    json.dump(data_dict, json_file, indent=4, sort_keys=True)
    json_file.close()
    return
