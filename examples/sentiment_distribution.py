import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_excel("topic_model_outputs/electric_vehicle_concat_24-Jan_3.xlsx")


# plt.figure(figsize=(10, 6))
# sns.boxenplot(x='dominant_topic', y='polarity', data=df, k_depth="proportion")
# plt.show()
fig, ax = plt.subplots(figsize=(9, 6))

sns.violinplot(x=df['inferred_topic'], y=df['polarity'], scale="count", inner="quartile", linewidthfloat=0.1, cut=0)
ax.set_title('Topic Sentiment Distribution', fontsize=15, loc='center')
ax.set_xlabel('Topics', fontsize=13)
ax.set_ylabel('Polarity', fontsize=13)

plt.tick_params(axis='y', which='major', labelsize=12)
plt.tick_params(axis='x', which='major', labelsize=12)
ax.yaxis.tick_left()  # where the y axis marks will be
plt.tight_layout()
plt.show()

"""Below is WIP code for magnitude topic sentiment bar charts."""


# ff = df[["inferred_topic", "status"]]

# topic_list = ff.inferred_topic.to_list()
# status_list = ff.status.to_list()
# d = {}
# for topic in topic_list:
#     d[topic] = [0, 0, 0]

# for topic, status in zip(topic_list, status_list):
#     if status == "pos":
#         d[topic][0] += 1
#     if status == "neutral":
#         d[topic][1] += 1
#     if status == "neg":
#         d[topic][2] += 1

# for i, v in d.items():
#     print(i, v)


# import numpy as np
# import matplotlib.pyplot as plt


# N = 5
# Pos = (1140, 287, 372, 943, 1055)
# Neut = (572, 140, 248, 332, 7)
# Neg = (308, 65, 32, 165, 1)
# ind = np.arange(N)
# width = 0.35

# p1 = plt.bar(ind, Pos, width, color='olivedrab')
# p2 = plt.bar(ind, Neut, width, color='orange')
# p3 = plt.bar(ind, Neg, width, color='firebrick')

# plt.ylabel('Number of Tweets')
# plt.xticks(ind, ('EV General', 'ICE Ban', 'Battery Technology', 'Charging', 'Nissan Sunderland'))
# plt.legend((p1[0], p2[0], p3[0]), ('Positive', 'Neutral', 'Negative'))

# plt.show()
# plt.show()
