import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

dummy_data = [
    "Test test test",
    "Hello hello hello.",
    "Hello my name is",
    "this is a test test test",
    "it is very rainy today",
    "the weather is poor today, it is very cold",
    "how are you today",
    "What is your name. My name is X"
]

df_dummy_data = pd.DataFrame(dummy_data, columns=['Test_Data'])


def get_top_x_ngrams(corpus, x=None, *, ngram_range=(3, 3)):
    """Extract top n-grams from dataframe column of texts.

    corpus : pandas.core.series.Series
        Column of text entries.
    x : int
        The number of n-grams to return (default is None).
    ngram_range : Iterable
        Iterable length 2, showing the range of n-grams to be returned.
        Default is (3, 3) (returning trigrams).

    """
    vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:x]


common_words = get_top_x_ngrams(df_dummy_data.Test_Data, 10)
trigrams = pd.DataFrame(common_words, columns=['trigram', 'count'])
print(trigrams)
