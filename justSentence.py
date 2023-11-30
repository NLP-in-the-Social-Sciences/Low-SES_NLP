
#Text vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
import numpy as np


listWords = ["school", "go", "job", "year", "work", "just", "time", "money", "take", "lot", "life", "know", "help", "college", "student", "family", "also", "think", "semester", "good", "tell", "graduate", "end", "friend", "point", "well", "live", "move", "need", "hard"]

# List of sentences, each representing a list of words
text = """my dad 's shop started until until latter part of my second year
I got decent paying job netting about 20 $ hour
I felt like my life was finally going somewhere
I used to move into apartment on school campus
I feel for first time actual drive to finish
My last semester stopped going altogether
her have decent stable jobs in new town
I done wrong compiled into my now life
I had just gotten out relationship
I really need advice from anyone
I working at my dad 's business
I went into slight depression
I was pushed to do in school
dead end job is in warehouse
I would only go on test days
I started skipping to point
she become much more caring
I was at During time school
me have stable jobs in town
I lived in freezing winter
I lived for almost 2 weeks
I lived before my dad let
I found something in life
I ended with sub 2.0 GPA
I making sort of money
I lost job because it
me feel at_time time
debt is with degree
I have friends
I do good
we screwed from over financially my stepdad 's misuse of our money
my step father attempted at_time suicide multiple times
I was already struggling as last few years of my life
I get airline ticket so as prize in lottery at work
I So booked trip right before fall semester started
our relatives kept as much at_time next day
I have attended three semesters of college
I So went to my former childhood therapist
I have attended as degree seeking student
I went to counseling center at university
life got again at_time even more so time
he So basically threatened to kill
I So emailed my academic counselor
we went to school full of stress
next semester would would great
he Finally was told to move out
he So basically threatened kill
I so continued my student job
us be good christian children
I called financial aid center
my Mom started Around time
he move after hospitalized
I went after breaking down
our relatives is in town
I thought for odd reason
I So emailed about it
money go to college
We go on boat ride
me was just lazy
I told myself
"""

#tfidf

preprocessed_text = text.lower()  # Convert to lowercase
sentences = preprocessed_text.split('\n')
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())


#Clustering Now
from hdbscan import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=5, min_samples=1)
clustering_result = clusterer.fit_predict(tfidf_matrix.toarray())

print(clustering_result)


#Quick check of the clusters

count  = 0
for l in clustering_result:

    count = count + 1

print(count)

#Clusters that were formed

# Create an empty dictionary to store clusters
clusters = {}

# Loop through each word and its corresponding cluster label
for word, cluster_label in zip(sentences, clustering_result):
    # Check if the word belongs to any cluster (i.e., not a noise point)\

     if cluster_label != -1:
        # Add the word to the corresponding cluster list in the dictionary
         if cluster_label not in clusters:
            clusters[cluster_label] = []
         clusters[cluster_label].append(word)

# Convert the dictionary of clusters to a list of lists
specific_clusters = list(clusters.values())

print(clusters)
print(len(sentences))

