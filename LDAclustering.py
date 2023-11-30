
#Text vectorization


from gensim.models import Word2Vec
import numpy as np


listWords = ["school", "go", "job", "year", "work", "just", "time", "money", "take", "lot", "life", "know", "help", "college", "student", "family", "also", "think", "semester", "good", "tell", "graduate", "end", "friend", "point", "well", "live", "move", "need", "hard"]


list_ListWords = [[words] for words in listWords]
# Create the Word2Vec model
model = Word2Vec(list_ListWords, vector_size=50, window=10, min_count=1, workers=5)

# Convert words to vectors
word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}

# Create a list of vectors for each word
vectors_list = [word_vectors[word] for sentence in list_ListWords for word in sentence]

# Convert the list of vectors into a numpy array
vectors_array = np.array(vectors_list)

print(vectors_array)




#Clustering Now
from hdbscan import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=3, min_samples=1)  # Adjust parameters as needed
clustering_result = clusterer.fit_predict(vectors_array)

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
for word, cluster_label in zip(list_ListWords, clustering_result):
    # Check if the word belongs to any cluster (i.e., not a noise point)\
    for w in word:
        if cluster_label != -1:
            # Add the word to the corresponding cluster list in the dictionary
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(w)

# Convert the dictionary of clusters to a list of lists
specific_clusters = list(clusters.values())

print(clusters)
