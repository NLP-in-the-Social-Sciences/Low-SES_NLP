# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:42:42 2022

@author: IJ
"""
import os
import re
import nltk
from nltk.corpus import wordnet as wn

file = open('topics2.txt','r') #file where the topics are stored
data = file.readlines()
list = []
set_all_topics = set()
synonyms_set = set()

#getting the topics from the topics file
for i in range(len(data)):
    data[i] = data[i].strip().split("\n")
    for j in range(len(data[i])):
        data[i][j] = re.sub(r'[^\w]', ' ', data[i][j])


    for el in data[i]:
        el = el.split()
        for el2 in el:
            if el2.isalpha():
                set_all_topics.add(el2)

#creating synonyms for the topics and adding them to the topics set
for el in set_all_topics:
    for syn in wn.synsets(el):
        for i in syn.lemmas():
            synonyms_set.add(i.name())
    
    
    
set_of_topics_with_synonyms = set_all_topics.union(synonyms_set)
filesList = os.listdir("filtered_triples//") # directory of the triples you want to compare
for i in range(len(filesList)):
    filetowrite = open("C:\\Users\\IJ\\Desktop\\research2023\\"+filesList[i]+"wordnet.txt","w") #directory where the triples will be stored
    file = open("filtered_triples//"+filesList[i], "r")
    for line in file:
        line2 = line.split()
        for el in line2:
            if el in set_of_topics_with_synonyms:
                filetowrite.write(line)
                break
    filetowrite.close()
    file.close()
    
print(set_of_topics_with_synonyms)   
    
    
