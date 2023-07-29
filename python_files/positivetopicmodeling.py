import pandas as pd
from gensim import corpora, models
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation 
nltk.download('punkt') 
nltk.download('stopwords') 
# Read the CSV file using pandas
reviews = pd.read_csv('data/BA_reviews_with_sentiment_and_category.csv', encoding='latin-1')

# Extract the text data from a specific column

positive_reviews = reviews[reviews['category'] == 'positive']
text_data = positive_reviews['reviews'].tolist() 

# Preprocess your text data (remove stopwords, punctuation, etc.) and tokenize it
stopwords_set = set(stopwords.words('english')) 
tokenized_documents = [] 

for document in text_data:
    tokens = word_tokenize(document.lower())
    tokens = [token for token in tokens if token not in stopwords_set and token not in punctuation]
    tokenized_documents.append(tokens)

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(tokenized_documents)

# Create a document-term matrix
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

# Train the LDA model
num_topics = 50
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

# Print the topics and associated keywords
for topic_id, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
    print(f"Topic ID: {topic_id}")
    for term, weight in topic:
        print(f"- {term}: {weight}")
    print()