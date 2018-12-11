from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

clusters = {}
word_to_cluster = {}

# with open("glove_clusters_50000.csv", encoding="utf-8") as cluster_id_file:
# 	for line in cluster_id_file:
# 		parts = line.strip().rsplit(",", 1)
# 		cluster_num = int(parts[1])
# 		if cluster_num in clusters:
# 			clusters[cluster_num].append(parts[0])
# 		else:
# 			clusters[cluster_num] = [parts[0]]
# 		word_to_cluster[parts[0]] = cluster_num

cluster_labels = {}
with open("vocab_clusters.csv", encoding="utf-8") as cluster_id_file:
	cluster_count = 0
	for line in cluster_id_file:
		parts = line.strip().split("\t")
		clusters[cluster_count] = ast.literal_eval(parts[1])
		cluster_labels[cluster_count] = parts[0]
		for w in clusters[cluster_count]:
			word_to_cluster[w] = cluster_count
		cluster_count += 1

cluster_colors = {}
for c in range(len(clusters)):
	cluster_colors[c] = np.random.rand(3,)

def tsne_plot(vectors, labels, clusters):
    "Creates and TSNE model and plots it"
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i], c=cluster_colors[clusters[i]])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

GLOVE_DIR = "../glove.6B/"
glove_vectors = []
glove_labels = []
glove_clusters = []

cluster_vectors = {i:[] for i in clusters}
glove_centroids = {i:np.zeros(200) for i in clusters}
print("reading in glove vectors...")
with open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'), encoding="utf-8") as f:
	count = 0
	for line in f:
		values = line.split()
		word = values[0]
		# print(count, word)
		if word in word_to_cluster:
			coefs = np.asarray(values[1:], dtype='float32')
			glove_vectors.append(coefs)
			glove_labels.append(word)
			cluster_id = word_to_cluster[word]
			glove_clusters.append(cluster_id)

			cluster_vectors[cluster_id].append((word, coefs))
			glove_centroids[cluster_id] += coefs
		count += 1
		if count >= 5000:
			break

# centroid_labels = {}
# for i in clusters:
# 	glove_centroids[i] /= len(cluster_vectors[i])
# 	closest_to_centroid = None
# 	min_dist = None
# 	for w, v in cluster_vectors[i]:
# 		dist = np.linalg.norm(v - glove_centroids[i])
# 		if closest_to_centroid is None or dist < min_dist:
# 			closest_to_centroid = w
# 			min_dist = dist
# 	centroid_labels[i] = closest_to_centroid

# with open("cluster_groups_50000.txt", "w", encoding="utf-8") as out_file:
# 	for cluster_num, cluster_list in clusters.items():
# 		out_file.write(str(cluster_num) + "\t" + centroid_labels[cluster_num] + "\t" + str(cluster_list) + "\n")

tsne_plot(glove_vectors, glove_labels, glove_clusters)