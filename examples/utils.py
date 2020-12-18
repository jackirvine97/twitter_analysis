"""Temporary collection of utility functions.
TODO - determine better utility strategy to avoid import errors."""


def search_past_7_days(search_term, max_tweets):
    """Returns specified number of tweets within the last 7 days.

    Parameters
    ----------
    search_term: :obj:`str`
        Search query.
    max_tweets: :obj:`int`
        Maximum number of tweets to be scraped.

    Returns
    -------
    :obj:`list`
        List of :obj:`tweepy.tweet` objects.

    """

    # Handle pagination using the cursor object.
    cursor = tweepy.Cursor(
        api.search,
        q=search_term,
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
