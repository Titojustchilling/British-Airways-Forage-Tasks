import matplotlib.pyplot as plt
import pandas as pd

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
reviews = pd.read_csv('data/BA_reviews_with_sentiment_and_category.csv', encoding='latin-1', usecols=['category'])
plt_df = reviews.category.value_counts().sort_values()
plt.bar(plt_df.index, plt_df.values)
addlabels(plt_df.index, plt_df.values)
plt.xlabel("Sentiment")
plt.ylabel("No. of Sentiment")
plt.title("Total # of Sentiment in Reviews")
plt.show()
