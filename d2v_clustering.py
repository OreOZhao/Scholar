import gensim
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec
from gensim.models.ldamodel import LdaModel
from util import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

lda_model = LdaModel.load('data/nat_title_topic.model')

nat_title_lda_probs = np.load('data/nat_title_lda_probs.npy')


def get_doc_topic_id(doc_id, docs_topics_prob):
    result = np.where(docs_topics_prob[doc_id] == docs_topics_prob[doc_id].max())
    return result[0][0]


LabeledSentence = gensim.models.doc2vec.TaggedDocument
nat_meta = read_processed_meta('data/nat_meta.hdf5')
docs = [' '.join(nat_meta[i].title) + ' ' + ' '.join(nat_meta[i].abstract) for i in range(len(nat_meta))]
content_train = []  # title + abstract
for i in range(len(docs)):
    content_train.append(LabeledSentence(docs[i], [i]))
    # # for multi tags
    # content_train.append(LabeledSentence(docs[i], [i, get_doc_topic_id(i, nat_title_lda_probs)]))

d2v_model = Doc2Vec(content_train, window=10, min_count=100, worker=8, dm=1, alpha=0.025, min_alpha=0.001)
d2v_model.train(content_train, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)

d2v_model.save('data/nat_doc2vec_model')

kmeans_model = KMeans(n_clusters=5, init='k-means++', max_iter=1000)
X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)

import matplotlib.pyplot as plt

fig = plt.figure()
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#77e0c6"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()

# EVALUATE CLUSTERING
# metrics.silhouette_score(d2v_lda_model.docvecs.doctag_syn0, labels, metric='euclidean')
# 轮廓系数 轮廓系数处于[-1,1]的范围内，-1表示错误的聚类，1表示高密度的聚类，0附近表示重叠的聚类；
# 0.064320594       # multi tags
# -0.007895615      $ single tags

# metrics.calinski_harabasz_score(d2v_lda_model.docvecs.doctag_syn0, labels)
# CH指标 当簇类密集且簇间分离较好时，Caliniski-Harabaz分数越高，聚类性能越好。
# 4406.480558416206     # multi tags
# metrics.silhouette_score(d2v_model.docvecs.doctag_syn0, labels, metric='euclidean')
# 419.0933658113983     # single tags

# UMAP Decrease Dimension

d2v = Doc2Vec.load('data/nat_doc2vec_model')
d2v_vecs = d2v.docvecs.doctag_syn0

import umap

umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(d2v_vecs[:1000])

# Birch Clustering
from sklearn.cluster import Birch

brc = Birch(branching_factor=50, n_clusters=5, threshold=0.1, compute_labels=True)
brc.fit(umap_data)
clusters = brc.predict(umap_data)
labels = brc.labels_.tolist()

fig = plt.figure()
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#77e0c6"]
color = [label1[i] for i in labels]
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=color)
plt.show()

from sklearn import metrics

metrics.silhouette_score(umap_data, labels)
# Birch 0.26647902

# AP Clustering
from sklearn.cluster import AffinityPropagation

af = AffinityPropagation(preference=-50).fit(umap_data)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)
plt.figure(figsize=(12, 8))
plt.title('Decomposition using UMAP')
plt.scatter(umap_data[:, 0], umap_data[:, 1])
plt.show()

# 0 cluster

# K-Means Clustering

kmeans_model = KMeans(n_clusters=5, init='k-means++', max_iter=1000)
kmeans_model.fit(umap_data)
labels = kmeans_model.labels_.tolist()
fig = plt.figure()
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#77e0c6"]
color = [label1[i] for i in labels]
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=color)
plt.show()
metrics.silhouette_score(umap_data, kmeans_model.labels_)


# 0.37112647

def doc_dist(a, b):
    return np.sqrt(sum(np.power((a - b), 2)))


def doc_sim(a, b):
    return 1 / (doc_dist(a, b) + 1)


labels = kmeans_model.labels_

cluster_to_doc = []
for i in range(5):
    cluster_to_doc.append([])
    cluster_to_doc[i] = [j for j in range(len(labels)) if labels[j] == i]

cen_id = []
for i in range(5):
    c = kmeans_model.transform(umap_data)[:, i]
    id = np.where(c == c.min())[0][0]
    cen_id.append(id)

nat_data = get_nature_data()
cen_doc = [nat_data.title[i] + " "+ nat_data.abstract[i] if nat_data.abstract[i] is not None else nat_data.title[i] for i in cen_id]

cluster_inner_sim = []
for i in range(5):
    l = len(cluster_to_doc[i])
    cluster_inner_sim.append(np.zeros([l, l]))
    for j in range(l):
        for k in range(l):
            a_id = cluster_to_doc[i][j]
            b_id = cluster_to_doc[i][k]
            cluster_inner_sim[i][j][k] = doc_sim(umap_data[a_id], umap_data[b_id])


def calc_ent(i, sim):
    m = len(sim[i])
    ent = 0
    for j in range(m):
        tmp = sim[i][j] * math.log2(sim[i][j] + 1)
        ent += tmp
    return ent


cluster_inner_ent = []
for i in range(5):
    l = len(cluster_to_doc[i])
    cluster_inner_ent.append(np.zeros(l))
    for j in range(l):
        cluster_inner_ent[i][j] = calc_ent(j, cluster_inner_sim[i])


highest_ent_id = [np.where(cluster_inner_ent[i] == cluster_inner_ent[i].max())[0][0] for i in range(5)]

highest_ent_doc_id = [cluster_to_doc[i][highest_ent_id[i]] for i in range(5)]

highest_ent_doc = [nat_data.title[i] + " "+ nat_data.abstract[i] if nat_data.abstract[i] is not None else nat_data.title[i] for i in highest_ent_doc_id]

# DBSCAN
from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=0.5)
y_pred = dbs.fit_predict(umap_data)
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=y_pred)
plt.show()
metrics.calinski_harabasz_score(umap_data, dbs.labels_)
metrics.silhouette_score(umap_data, dbs.labels_)
