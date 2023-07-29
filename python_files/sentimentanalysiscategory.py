import matplotlib.pyplot as plt
import pandas as pd


reviews = pd.read_csv('data/BA_reviews_with_sentiment.csv', encoding='latin-1')
text_data = reviews['sentiment']

category = []

for sentiment in text_data:

    # Append the sentiment score to the list
    if sentiment > 0: 
        sentiment = 'positive' 
        category.append(sentiment)
    elif sentiment == 0: 
        sentiment = 'neutral'
        category.append(sentiment) 
    elif sentiment < 0: 
        sentiment = 'negative'
        category.append(sentiment)

reviews['category'] = category

# Save the DataFrame to a new CSV file
reviews.to_csv('data/BA_reviews_with_sentiment_and_category.csv', index=False)