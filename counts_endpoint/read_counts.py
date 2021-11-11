from datetime import datetime
import json
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = 'Arial'


json_filename = "facebook.json"
with open(json_filename) as json_file:
    facebook_dict = json.load(json_file)

json_filename = "whatsapp.json"
with open(json_filename) as json_file:
    whatsapp_dict = json.load(json_file)

json_filename = "instagram.json"
with open(json_filename) as json_file:
    instagram_dict = json.load(json_file)

tweet_count_list = [fb["tweet_count"] + wa["tweet_count"] + ig["tweet_count"] for fb, wa, ig in zip(facebook_dict["data"], whatsapp_dict["data"], instagram_dict["data"])]
timestamp_list = [fb["start"] for fb in facebook_dict["data"]]

df = pd.DataFrame(timestamp_list)
df["Count"] = tweet_count_list
df.columns = ["timestamp", "count"]
df = df.sort_values("timestamp").reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Build plot
fig, ax = plt.subplots()

# Shade background during outage
outage_start = "10/04/21 15:39:00"
outage_end = "10/04/21 22:50:00"
ax.axvspan(datetime.strptime(outage_start, "%x %X"), datetime.strptime(outage_end, "%x %X"), alpha=0.1, color='tab:grey', label="6 Hour Global Outage")

ax.plot(df["timestamp"], df["count"], color="deepskyblue")
ax.set_xlabel("Time (UTC), 04/10/2021 to 05/10/2021", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of Tweets Per Minute", fontsize=11, fontweight="bold")
x_format = mdates.DateFormatter('%H:%M:%S  ax.xaxis.set_major_formatter(x_format)  # Format x-axis
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))  # Add major x-ticks
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30], interval=1))  # Add minor x-ticks every 30 minutes
ax.set_ylim(0, 81000)


# Annotation metadata
an_color = "steelblue"
point_size = 4

# Annotate chart with an event
event_time = "10/04/21 16:16:00"
event_point = 15000
label = '16:16 UTC: WhatsApp\nacknowledge outage'
text_pos_rel_point = (-75, 50)
ax.vlines(datetime.strptime(event_time, "%x %X"), 0, event_point, linestyle=":", color=an_color)
ax.plot(datetime.strptime(event_time, "%x %X"), event_point, "-o", color=an_color, markersize=point_size)
an1 = ax.annotate(
    label,
    fontsize=9,
    xy=(datetime.strptime(event_time, "%x %X"), event_point + 25),
    xytext=text_pos_rel_point,
    textcoords='offset points',
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2")
)

# Annotate chart with an event
event_time = "10/04/21 16:22:00"
event_point = 15000
label = '16:22 UTC: Facebook\nacknowledge outage'
text_pos_rel_point = (-75, 50)
ax.vlines(datetime.strptime(event_time, "%x %X"), 0, event_point, linestyle=":", color=an_color)
ax.plot(datetime.strptime(event_time, "%x %X"), event_point, "-o", color=an_color, markersize=point_size)
an2 = ax.annotate(
    label,
    fontsize=9,
    xy=(datetime.strptime(event_time, "%x %X"), event_point + 25),
    xytext=text_pos_rel_point,
    textcoords='offset points',
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2")
)

# Annotate chart with an event
event_time = "10/04/21 16:44:00"
event_point = 15000
label = '16:44 UTC: BBC\nreport outage via\n@BBCBreaking Twitter handle'
text_pos_rel_point = (-75, 50)
ax.vlines(datetime.strptime(event_time, "%x %X"), 0, event_point, linestyle=":", color=an_color)
ax.plot(datetime.strptime(event_time, "%x %X"), event_point, "-o", color=an_color, markersize=point_size)
an3 = ax.annotate(
    label,
    fontsize=9,
    xy=(datetime.strptime(event_time, "%x %X"), event_point + 25),
    xytext=text_pos_rel_point,
    textcoords='offset points',
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2")
)

# Annotate chart with an event
event_time = "10/04/21 17:06:00"
event_point = 15000
label = '17:06 UTC: CNN\nreport outage via\n@CNNbrk Twitter handle'
text_pos_rel_point = (-75, 50)
ax.vlines(datetime.strptime(event_time, "%x %X"), 0, event_point, linestyle=":", color=an_color)
ax.plot(datetime.strptime(event_time, "%x %X"), event_point, "-o", color=an_color, markersize=point_size)
an4 = ax.annotate(
    label,
    fontsize=9,
    xy=(datetime.strptime(event_time, "%x %X"), event_point + 25),
    xytext=text_pos_rel_point,
    textcoords='offset points',
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
)

# Annotate chart with an event
event_time = "10/04/21 17:26:00"
event_point = 15000
label = '17:26 UTC: CBS\nreport outage via\n@CBSNews Twitter handle'
text_pos_rel_point = (-75, 50)
ax.vlines(datetime.strptime(event_time, "%x %X"), 0, event_point, linestyle=":", color=an_color)
ax.plot(datetime.strptime(event_time, "%x %X"), event_point, "-o", color=an_color, markersize=point_size)
an5 = ax.annotate(
    label,
    fontsize=9,
    xy=(datetime.strptime(event_time, "%x %X"), event_point + 25),
    xytext=text_pos_rel_point,
    textcoords='offset points',
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
)

# Annotate chart with an event
event_time = "10/04/21 22:05:00"
event_point = 15000
label = '22:05 UTC: Facebook domain\nname system servers\nbegin to become available'
text_pos_rel_point = (-75, 50)
ax.vlines(datetime.strptime(event_time, "%x %X"), 0, event_point, linestyle=":", color=an_color)
ax.plot(datetime.strptime(event_time, "%x %X"), event_point, "-o", color=an_color, markersize=point_size)
an6 = ax.annotate(
    label,
    fontsize=9,
    xy=(datetime.strptime(event_time, "%x %X"), event_point + 25),
    xytext=text_pos_rel_point,
    textcoords='offset points',
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
)

# Annotate chart with an event
event_time = "10/04/21 19:52:00"
event_point = 15000
label = '19:52 UTC: Facebook CTO,\nMike Schroepfer, acknowledges\n\'network issues\' and apologises'
text_pos_rel_point = (-75, 50)
ax.vlines(datetime.strptime(event_time, "%x %X"), 0, event_point, linestyle=":", color=an_color)
ax.plot(datetime.strptime(event_time, "%x %X"), event_point, "-o", color=an_color, markersize=point_size)
an7 = ax.annotate(
    label,
    fontsize=9,
    xy=(datetime.strptime(event_time, "%x %X"), event_point + 25),
    xytext=text_pos_rel_point,
    textcoords='offset points',
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", fc="0.8"),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
)

# Make annotation draggable
an1.draggable()
an2.draggable()
an3.draggable()
an4.draggable()
an5.draggable()
an6.draggable()
an7.draggable()

ax.legend(fontsize=11)

ax.text(x=datetime.strptime("10/04/21 12:00:00", "%x %X"), y=90000, s="Following the Global Facebook Outage, 4th October 2021, millions took to Twitter to report\nand discuss the event",
        fontsize=13, weight='bold')
ax.text(x=datetime.strptime("10/04/21 12:00:00", "%x %X"), y=85000,
        s='Number of tweets worldwide by the minute mentioning "Facebook", "Instagram" or "WhatsApp"',
        fontsize=9, alpha=.85)

#plt.suptitle('Following the Global Facebook Outage on 4th october 2021, millions took to Twitter to report the outage', fontsize=15, fontweight="bold")
#plt.title('Number of Tweets by the minute mentioning "Facebook", "Instagram" or "WhatsApp"', fontsize=11, loc="left")

plt.tight_layout()
plt.show()


"""

Labels - facebook official announcement 4:22 UTC
	   - BBC breaking tweet: 4:44 UTC
	   - sky news 4:28
	   - 5:06 cnn breaaking
	   - cbs news 5:26
	   - 22:05 - domain name services available
	   - 22:50 - generally available
	   - 4:25 instagram
	   - 4:16 whatsapp


#curl --request GET 'https://api.twitter.com/2/tweets/counts/recent?query=(facebook OR whatsapp OR instagram)&granularity=minute&start_time=2021-10-04T14:00:00Z&end_time=2021-10-05T02:00:00Z' --header 'Authorization: Bearer AAAAAAAAAAAAAAAAAAAAAAE6JgEAAAAAPPVgLAS%2BDbDRLqcEAJHXrenlsec%3Dj2dUMI7ch6nscxWkmC4q9jHBOTGVBjRKKKJa6h0BPQB6vlalz3' -o facebook.json   
#curl --request GET 'https://api.twitter.com/2/tweets/counts/recent?query=facebook&granularity=minute&start_time=2021-10-04T13:00:00Z&end_time=2021-10-05T02:00:00Z' --header 'Authorization: Bearer AAAAAAAAAAAAAAAAAAAAAAE6JgEAAAAAPPVgLAS%2BDbDRLqcEAJHXrenlsec%3Dj2dUMI7ch6nscxWkmC4q9jHBOTGVBjRKKKJa6h0BPQB6vlalz3' -o facebook.json 

"""