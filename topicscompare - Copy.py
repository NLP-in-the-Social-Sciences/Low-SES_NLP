# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:42:42 2022

@author: IJ
"""
import os
import re
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
    




file = open('topics2.txt','r') #file where the topics are stored
data = file.readlines()
list = []
set_all_topics = []
#synonyms_set = set()

#getting the topics from the topics file
for i in range(len(data)):
    data[i] = data[i].strip().split("\n")
    for j in range(len(data[i])):
        data[i][j] = re.sub(r'[^\w]', ' ', data[i][j])


    for el in data[i]:
        el = el.split()
        for el2 in el:
            if el2.isalpha():
                set_all_topics.append(el2)
                
print(set_all_topics)

#creating synonyms for the topics and adding them to the topics set
#for el in set_all_topics:
#    for syn in wn.synsets(el):
#        for i in syn.lemmas():
#            synonyms_set.add(i.name())


    
filesList = os.listdir("filtered_triples//") # directory of the triples you want to compare
embedding_of_topics = model.encode(set_all_topics, convert_to_tensor=True)

for i in range(len(filesList)):
    filetowrite = open("C:\\Users\\IJ\\Desktop\\research2023\\newversion\\"+filesList[i]+"wordnet.txt","w") #directory where the triples will be stored
    file = open("filtered_triples//"+filesList[i], "r")
    for line in file:
        line2 = line.split()
        for el in line2:
            found_similar_word = False
            word_embedding = model.encode(el, convert_to_tensor=True)
            #if el in set_of_topics_with_synonyms:
            for i, w2 in enumerate(set_all_topics):
                # Compute cosine-similarity between embeddings
                cosine_score = util.cos_sim(word_embedding, embedding_of_topics[i]).item()
                # If score is greater than 0.65, print the words and score and move to next word in word1
                if cosine_score > 0.65:
                    found_similar_word = True
                    filetowrite.write(line)
                    break
            if found_similar_word:
                break
    filetowrite.close()
    file.close()
    
   
    
    
