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


dfs_1 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-31-Jan-2021.json")[0] for index in range(1, 5)]
dfs_2 = [open_json_as_dataframe(f"../data/ICE_ban_November_2020_{index}-01-Feb-2021.json")[0] for index in range(5, 10)]
df = pd.concat(dfs_1 + dfs_2)

df = df.loc[df["rt"] is False]  # Filter out RTs

tweets_list = df.text.to_list()
tweets_list = clean(tweets_list)
common_words = get_top_x_ngrams(tweets_list, 50)
trigrams = pd.DataFrame(common_words, columns=['trigram', 'count'])

# Plot most frequent trigrams.
plt.figure()
plt.barh(trigrams.trigram, trigrams["count"])
plt.xlabel("Top Trigram")
plt.ylabel("Number of Tweets")
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