
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


x = "i hate this"

vs = analyzer.polarity_scores(x)
print(vs["compound"])

