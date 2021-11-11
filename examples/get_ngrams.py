import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils import clean, open_json_as_dataframe


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


# df = pd.read_pickle("../processed_data/EVs_2020_Tweets_sent_topics.pkl")
# df = df.loc[df["rt"]==False]  # Filter out RTs
# df = df[~df['text'].str.contains('hr6201')]
# df = df[~df['text'].str.contains('HR6201')]
# df = df[~df['text'].str.contains('JohnKingCNN')]
# df = df[~df['text'].str.contains('Started a space company &gt')]
# df = df[~df['text'].str.contains('CraigCaplan')]
# df = df[~df['text'].str.contains('Shane_Evs')]
# df = df[~df['text'].str.contains('Evs_Dubai')]

df = pd.read_pickle(f"../processed_data/EVs_2020_Tweets_sent_2.pkl")
df = df.loc[df["rt"]==False]  # Filter out RTs
df = df[~df['text'].str.contains('@fkabudu')]
df = df[~df['text'].str.contains('jaguar')]

tweets_list = df.text.to_list()
tweets_list = clean(tweets_list)
common_words = get_top_x_ngrams(tweets_list, 30, ngram_range=(3, 3))
trigrams = pd.DataFrame(common_words, columns=['trigram', 'count'])

# Plot most frequent trigrams.
plt.figure()
plt.barh(trigrams.trigram, trigrams["count"])
plt.ylabel("Top Trigram")
plt.xlabel("Number of Tweets")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()





"""
Create trigram network plot.
"""

# for row_index in trigrams.index:
#     trigrams.at[row_index, "trigram"] = tuple(trigrams["trigram"][row_index].split())

# # Create dictionary of bigrams and their counts
# trigram_dict = trigrams.set_index('trigram').T.to_dict('records')

# # Create network plot
# G = nx.Graph()

# # Create connections between nodes
# for k, v in trigram_dict[0].items():
#     G.add_edge(k[0], k[1], weight=(v * 10))

# fig, ax = plt.subplots(figsize=(10, 8))

# pos = nx.spring_layout(G, k=15)

# # Plot networks
# nx.draw_networkx(G, pos,
#                  font_size=9,
#                  width=1,
#                  edge_color='grey',
#                  node_color='purple',
#                  with_labels=False,
#                  ax=ax)

# # Create offset labels
# for key, value in pos.items():
#     x, y = value[0]+0.1, value[1]+0.03
#     ax.text(x, y,
#             s=key,
#             bbox=dict(facecolor='red', alpha=0.25),
#             horizontalalignment='center', fontsize=9)
    
# plt.show()