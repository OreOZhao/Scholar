import gensim
from gensim.models.doc2vec import Doc2Vec
from util import *
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

LabeledSentence = gensim.models.doc2vec.TaggedDocument
nat_meta = read_processed_meta('data/nat_meta.hdf5')
docs = [' '.join(nat_meta[i].title) + ' ' + ' '.join(nat_meta[i].abstract) for i in range(len(nat_meta))]
content_train = []  # title + abstract
for i in range(len(docs)):
    content_train.append(LabeledSentence(docs[i], [i]))

d2v_model = Doc2Vec(content_train, window=10, min_count=500, worker=8, dm=1, alpha=0.025, min_alpha=0.001)
d2v_model.train(content_train, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)

d2v_model.save('data/nat_doc2vec_model')

kmeans_model = KMeans(n_clusters=5, init='k-means++', max_iter=1000)
X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)

import matplotlib.pyplot as plt

fig = plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#77e0c6"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()
