from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Prepare your text data
reviews = pd.read_csv('data/BA_reviews.csv', encoding='latin-1')

# Convert the reviews column to a single string
text_data = ' '.join(reviews['reviews'])

# Create a word cloud from the text data
cloud = WordCloud().generate(text_data)

# Display the generated word cloud
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()