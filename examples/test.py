#     # Tabulate topic distribution for each document.
#     topic_distribution_per_doc_dict = {}
#     for doc_topic_disttribution in lda_model.get_document_topics(corpus):
#         for topic_id, probability in doc_topic_disttribution:
#             topic_distribution_per_doc_dict.setdefault(topic_id, []).append(probability)
#     # topic_distribution_per_doc_df = pd.DataFrame(topic_distribution_per_doc_dict)
#     # topic_distribution_per_doc_df.to_excel("topic_distribution_per_doc.xlsx")
#     for k, v in topic_distribution_per_doc_dict.items():
#         print(len(v))

# The above aims to get the distribution of topics for each document. It fails as the length of values for each topic in the dict varies topic to topic.

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# x = np.linspace(0, 10, 100)
# lines = [m*x for m in range(26)]

# evenly_spaced_interval = np.linspace(0, 1, len(lines))
# colors = [matplotlib.cm.tab20(x) for x in range(26)]

# print(len(colors))
# for i, color in enumerate(colors):
#     plt.plot(lines[i], color = color)

# plt.show()
import pandas as pd
df = pd.DataFrame([{"A": "88", "B": 1}, {"A": "88", "B": 1}])
df.index.name = 'hi'
print(df)
# df.rename(columns={"A": "a", "B": "c"}, inplace=True)
# df.drop('c', axis=1, inplace=True)
# df.to_pickle('test.pkl')
# x = pd.read_pickle('test.pkl')
# print(x)