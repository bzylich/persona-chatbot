__author__ = "Brian Zylich"

from sklearn.cluster import KMeans
import numpy as np
import os

def get_closest_word(collection, labels, vector):
	min_dist = None
	min_word = None
	for c in range(len(collection)):
		dist = np.linalg.norm(np.array(collection[c]) - np.array(vector))
		if min_dist is None or dist < min_dist:
			min_dist = dist
			min_word = labels[c]
	return min_word

GLOVE_DIR = "../glove.6B/"
glove_vocab = []
glove_labels = []
print("reading in glove vectors...")
with open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'), encoding="utf-8") as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		glove_vocab.append(coefs)
		glove_labels.append(word)

desired_vocab_size = 8000
print("training with kmeans...")
kmeans = KMeans(n_clusters=desired_vocab_size).fit(np.array(glove_vocab))
glove_cluster_labels = list(kmeans.labels_)
glove_centroids = list(kmeans.cluster_centers_)

print("writing glove cluster mappings to file...")
with open("glove_clusters.csv", "w", encoding="utf-8") as glove_cluster_file:
	for i in range(len(glove_cluster_labels)):
		glove_cluster_file.write(glove_labels[i] + "," + str(glove_cluster_labels[i]) + "\n")

glove_centroid_labels = {}

print("finding words associated with centroids...")
for i in range(len(glove_centroids)):
	glove_centroid_labels[i] = get_closest_word(glove_vocab, glove_labels, glove_centroids[i])

print("writing out cluster centroids to file...")
with open("cluster_centroids.csv", "w", encoding="utf-8") as cluster_centroid_file:
	count = 0
	for cluster, word in glove_centroid_labels.items():
		cluster_centroid_file.write(word + "," + str(cluster) + "," + str(list(glove_centroids[count])) + "\n")
		count += 1