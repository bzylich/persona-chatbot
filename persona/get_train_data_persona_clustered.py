# -*- coding: utf-8 -*-
# Modified by: Albert Enyedy and Brian Zylich

# uses clustered word embeddings to create a vocabulary of 8000 words and a mapping of 
# almost 50K words to that set of 8000, then vectorizes data for training

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import numpy as np
# np.random.seed(1234)  # for reproducibility
import pandas as pd
import os
import csv
import nltk
import itertools
import operator
import pickle
import numpy as np    
from keras.preprocessing import sequence
from scipy import sparse, io
from numpy.random import permutation
import re
import ast
    
questions_file = 'context'
answers_file = 'answers'
personas_file = 'personas'
vocabulary_file = 'vocabulary_personas_clustered_test'
padded_questions_file = 'Padded_context_clustered'
padded_answers_file = 'Padded_answers_clustered'
padded_personas_file = 'Padded_personas_clustered'
unknown_token = 'something'

vocabulary_size = 7999
max_features = vocabulary_size
maxlen_input = 50
maxlen_output = 50  # cut texts after this number of words

print ("Reading the context data...")
q = open(questions_file, 'r')
questions = q.read()
print ("Reading the persona data...")
p = open(personas_file, 'r')
personas = p.read()
print ("Reading the answer data...")
a = open(answers_file, 'r')
answers = a.read()
all = answers + personas + questions
print ("Tokenizing the answers...")
paragraphs_a = [p for p in answers.split('\n')] # answer lines
paragraphs_b = [p for p in all.split('\n')] # all lines
paragraphs_a = ['BOS '+p+' EOS' for p in paragraphs_a] # answer lines with BOS and EOS
paragraphs_b = ['BOS '+p+' EOS' for p in paragraphs_b] # all lines with BOS and EOS
paragraphs_b = ' '.join(paragraphs_b) # all lines w/ BOS and EOS concatenated as string
tokenized_text = paragraphs_b.split() # all words (including BOS and EOS tokens)
paragraphs_q = [p for p in questions.split('\n') ] # question lines
paragraphs_persona = [p for p in personas.split('\n')] # persona lines
tokenized_answers = [p.split() for p in paragraphs_a]
tokenized_questions = [p.split() for p in paragraphs_q]
tokenized_personas = [p.split() for p in paragraphs_persona]

# Counting the word frequencies:
word_freq = nltk.FreqDist(itertools.chain(tokenized_text))
all_vocab_words = set(map(lambda i: i[0], word_freq.items()))
print ("Found %d unique words tokens." % len(word_freq.items()))

clusters = []
default_cluster_labels = []
cluster_labels = []
word_to_cluster = {}
vocab_count = 0
pruned_vocab_count = 0
with open("cluster_groups_50000.txt", "r", encoding="utf-8") as cluster_file:
    count = 0
    for line in cluster_file:
        parts = line.strip().split("\t")
        cluster_words = ast.literal_eval(parts[2])
        vocab_count += len(cluster_words)
        if len(cluster_words) > 2:
            clusters.append(cluster_words)
            default_cluster_labels.append(parts[1])
            cluster_labels.append(None)
            for w in cluster_words:
                word_to_cluster[w] = count
                pruned_vocab_count += 1
            count   += 1

print("original vocab size", vocab_count, "pruned vocab size", pruned_vocab_count)
print("num clusters", len(clusters))

# Getting the most common words and build index_to_word and word_to_index vectors:
vocab = word_freq.most_common(vocabulary_size)


missing_count = 0
for w in vocab:
    if w[0] not in word_to_cluster:
        # print(w, "not in clusters")
        clusters.append([w[0]])
        cluster_labels.append(w[0])
        word_to_cluster[w[0]] = len(clusters) - 1
        missing_count += 1
print("clusters miss", missing_count, "words")
print("num clusters", len(clusters))
print("new vocab size", pruned_vocab_count + missing_count)


for i in range(vocabulary_size):
    most_common_word = vocab[i][0]
    cluster_index = word_to_cluster[most_common_word]
    cluster_label = cluster_labels[cluster_index]
    if cluster_label is None:
        cluster_labels[cluster_index] = most_common_word

i = 0
while len(clusters) < vocabulary_size and i < vocabulary_size:
    most_common_word = vocab[i][0]
    cluster_index = word_to_cluster[most_common_word]
    cluster_label = cluster_labels[cluster_index]
    if cluster_label is not None and cluster_label != most_common_word:
        # print(most_common_word, clusters[cluster_index])
        clusters[cluster_index].remove(most_common_word)
        cluster_index = len(clusters)
        word_to_cluster[most_common_word] = cluster_index
        clusters.append([most_common_word])
        cluster_labels.append(most_common_word)
    i += 1

for i in range(len(clusters)):
    if cluster_labels[i] is None:
        cluster_labels[i] = default_cluster_labels[i]

vocab = list(map(lambda l: (l, 0), cluster_labels))

print("num clusters", len(clusters), "cluster labels", len(cluster_labels))

expanded_vocab_size = 0
with open("vocab_clusters_test.csv", "w", encoding="utf-8") as out_file:
    for i in range(len(clusters)):
        expanded_vocab_size += len(clusters[i])
        out_file.write(cluster_labels[i] + "\t" + str(clusters[i]) + "\n")
        for w in clusters[i]:
            if w in all_vocab_words:
                all_vocab_words.remove(w)
print("expanded vocab size", expanded_vocab_size)
print("missed", len(all_vocab_words), "words from complete vocab")
with open("missed_vocab_persona.csv", "w", encoding="utf-8") as missed_file:
    for missed in all_vocab_words:
        missed_file.write(missed + "\n")

# Saving vocabulary:
with open(vocabulary_file, 'wb') as v:
   pickle.dump(vocab, v)

vocab = pickle.load(open(vocabulary_file, 'rb'))


index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print ("Using vocabulary of size %d." % vocabulary_size)
# print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replacing all words not in our vocabulary with the unknown token:
for i, sent in enumerate(tokenized_answers):
    tokenized_answers[i] = [cluster_labels[word_to_cluster[w]] if w in word_to_cluster else unknown_token for w in sent]
   
for i, sent in enumerate(tokenized_questions):
    tokenized_questions[i] = [cluster_labels[word_to_cluster[w]] if w in word_to_cluster else unknown_token for w in sent]

for i, sent in enumerate(tokenized_personas):
    tokenized_personas[i] = [cluster_labels[word_to_cluster[w]] if w in word_to_cluster else unknown_token for w in sent]

# Creating the training data:
X = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_questions])
Y = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_answers])
Z = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_personas])

Q = sequence.pad_sequences(X, maxlen=maxlen_input)
P = sequence.pad_sequences(Z, maxlen=maxlen_input)
A = sequence.pad_sequences(Y, maxlen=maxlen_output, padding='post')

with open(padded_questions_file, 'wb') as q:
    pickle.dump(Q, q)
    
with open(padded_answers_file, 'wb') as a:
    pickle.dump(A, a)

with open(padded_personas_file, 'wb') as p:
    pickle.dump(P, p)