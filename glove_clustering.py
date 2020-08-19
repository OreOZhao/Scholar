import codecs
import json
from numbers import Number
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from util import *

import snap

# Use K-Means to build cluster of GloVe word vector in title.

# unprocessed data
# meta_list = read_meta('origin_data.hdf5')
# data = get_data()
# data = data.reset_index(drop=True)
# for i in range(len(meta_list)):
#     if isinstance(data.citation[i], str):
#         meta_list[i].citation = data.citation[i].replace("['", '').replace("']", '').split("', '")
#     else:
#         meta_list[i].citation = None
# sci_meta = []
# nat_meta = []
# for m in meta_list:
#     if m.citation is None:
#         sci_meta.append(m)
#     else:
#         nat_meta.append(m)

stop_words = read_set_from_file('data/stopwords.txt')
vector_file = 'GloVe/nat_title_vectors.txt'
corpus = get_corpus_from_file('data/nat_title_origin_corpus.txt')
corpus = [corpus[i] for i in range(len(corpus)) if corpus[i] not in stop_words]


# nat_data = get_nature_data()
# nat_abstract = nat_data.abstract.to_list()
# nat_abstract = [a for a in nat_abstract if a is not None]


def get_origin_abstract_corpus(nat_abstract):
    s = set([])
    for i in range(len(nat_abstract)):
        a = re.split('[ @$/#.-:&*+=\[\]?!()\{\},\'\">_<;%]', nat_abstract[i].lower())
        temp = []
        for j in a:
            if len(j) > 1 and '-' not in j and '–' not in j and '—' not in j:
                temp.append(j)
        s = s.union(set(temp))
    write_set_to_file('nat_abstract_origin_corpus.txt', s)


def get_origin_title_corpus(meta_list):
    s = set([])
    for m in meta_list:
        title = re.split('[ @$/#.-:&*+=\[\]?!()\{\},\'\">_<;%]', m.title.lower())
        temp = []
        for i in title:
            if len(i) > 1 and '-' not in i and '–' not in i and '—' not in i:
                temp.append(i)
        s = s.union(set(temp))
    write_set_to_file('nat_title_origin_corpus.txt', s)
    return s


def build_word_vector_matrix(corpus, vector_file):
    vec_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        # labels_array = [line.split()[0] for line in enumerate(f) if
        #                 len(line.split()) == 51 and line.split()[0] in corpus]
        for i, line in enumerate(f):
            sr = line.split()
            if sr[0] in corpus and len(sr) == 51:
                labels_array.append(sr[0])
                vec_arrays.append(np.array([float(j) for j in sr[1:]]))
    return np.array(vec_arrays), labels_array


def find_word_clusters(label_array, cluster_label):
    cluster_to_word = []
    for i in range(100):
        cluster_to_word.append([])
        cluster_to_word[i] = [label_array[j] for j in range(len(label_array)) if cluster_label[j] == i]
    return cluster_to_word


def save_json(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


vec_array, label_array = build_word_vector_matrix(corpus, vector_file)

# kmeans_model = KMeans(init='k-means++', n_clusters=100, n_jobs=10, n_init=10)
# vec_arrays = np.array(vec_arrays)
# kmeans_model.fit(vec_arrays)
# cluster_labels = kmeans_model.labels_
# cluster_to_words = list(find_word_clusters(labels_array, cluster_labels).values())
# save_json('data/nat_abstract_50D_100_clusters.json', cluster_to_words)
#
# # Abstract
# # pretrained cluster
# # metrics.calinski_harabasz_score(vec_arrays, cluster_labels)
# # Out[17]: 255.53576999843693
# # trained cluster
# # metrics.calinski_harabasz_score(nat_vec_arrays, cluster_labels)
# # Out[23]: 112.58887990127593
#
# # Title
# # pretrained
# # CH score  119.45812175609963
# # silhouette score 0.020828687425234846
# # newly trained
# # CH score
# # silhouette score
#
# nat_pca = PCA(n_components=2).fit(nat_vec_arrays)
# nat_datapoints = nat_pca.transform(nat_vec_arrays)
# nat_pca_kmeans_model = KMeans(init='k-means++', n_clusters=30, n_jobs=10, n_init=10)
# nat_pca_kmeans_model.fit(nat_datapoints)
#
# nat_pca_cluster_labels = nat_pca_kmeans_model.labels_
# metrics.silhouette_score(nat_datapoints, nat_pca_cluster_labels, metric='euclidean')
# ch_score = []
# sil_score = []
#
# for nc in list([2, 5, 10, 20, 30, 40, 50]):
#     pca = PCA(n_components=nc).fit(vec_arrays)
#     datapoints = pca.transform(vec_arrays)
#     kmeans_model = KMeans(init='k-means++', n_clusters=100, n_jobs=10, n_init=10)
#     kmeans_model.fit(datapoints)
#     cluster_labels = kmeans_model.labels_
#     ch = metrics.calinski_harabasz_score(datapoints, cluster_labels)
#     ss = metrics.silhouette_score(datapoints, cluster_labels, metric='euclidean')
#     ch_score.append(ch)
#     sil_score.append(ss)
#
# # [0.3211418747332224,
# #  0.12247357596294567,
# #  0.04646632446847809,
# #  -0.001388342985954234,
# #  -0.024015410242636457,
# #  -0.03058109498397773]
#
# pre_vec_arrays2 = np.array([pre_vec_arrays[i] for i in range(len(pre_vec_arrays)) if pre_labels_array[i] in labels_array])
# pre_label_array2 = [pre_labels_array[i] for i in range(len(pre_labels_array)) if pre_labels_array[i] in labels_array]
#
# con_vec_arrays = np.concatenate((pre_vec_arrays2[0], vec_arrays[5896]))
# for i in range(1, len(pre_vec_arrays2)):
#     term = pre_label_array2[i]
#     j = labels_array.index(term)
#     row = np.concatenate((vec_arrays[j], pre_vec_arrays2[i]))
#     con_vec_arrays = np.row_stack((con_vec_arrays, row))
#
# con_vec_array = con_vec_arrays
# con_label_array = pre_label_array2
#
# pca = PCA(n_components=2).fit(con_vec_array)
# datapoints = pca.transform(con_vec_array)

# kmeans_model = KMeans(init='k-means++', n_clusters=100, n_jobs=10, n_init=10)
# kmeans_model.fit(con_vec_array)
# cluster_to_words = list(find_word_clusters(con_label_array, cluster_labels).values())
# save_json('data/glove/nat_abstract_newly_pre_con_100D_100_clusters.json', cluster_to_words)
# cluster_labels = kmeans_model.labels_
# metrics.calinski_harabasz_score(con_vec_array, cluster_labels)
# metrics.silhouette_score(con_vec_array, cluster_labels)

# metrics.calinski_harabasz_score(con_vec_array, cluster_labels)
# Out[77]: 86.40351081125746
# metrics.silhouette_score(con_vec_array, cluster_labels)
# Out[78]: 0.010331794489210485

# ch_score = []
# sil_score = []
# for nc in list([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
#     pca_km_model = KMeans(init='k-means++', n_clusters=nc, n_jobs=10, n_init=10)
#     pca_km_model.fit(datapoints)
#     pca_cluster_labels = pca_km_model.labels_
#     ch = metrics.calinski_harabasz_score(datapoints, pca_cluster_labels)
#     ss = metrics.silhouette_score(datapoints, pca_cluster_labels, metric='euclidean')
#     ch_score.append(ch)
#     sil_score.append(ss)
# ch_score
# [16830.33902739976,
#  16104.242323435368,
#  15718.324786115034,
#  15458.553409004135,
#  15325.81514874963,
#  15348.84176319364,
#  15305.865597035576,
#  15247.147884686327,
#  15251.508765890558,
#  15182.556593531412]
#
#
# sil_score
# [0.3334981763366016,
#  0.3265813884865134,
#  0.3225273206200643,
#  0.3163097755219793,
#  0.3192230639069943,
#  0.3205364585548827,
#  0.321565983009238,
#  0.3218568178162822,
#  0.32322619547374637,
#  0.32317695985015815]

# K-Means model With Sample Weight
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

title = get_corpus_from_file('data/nat_title.txt')
for i in range(len(title)):
    tmp = title[i].lower().split()
    tmp = [tmp[j] for j in range(len(tmp)) if tmp[j] in corpus]
    title[i] = ' '.join(tmp)

# from sklearn.feature_extraction.text import TfidfVectorizer
#
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(title)

full_title = []
for i in range(len(title)):
    full_title.extend(title[i].split())

vec_array = np.load('data/nat_con_vector.npy')
label_array = np.load('data/nat_con_vec_label.npy').tolist()

freq_weight = [full_title.count(word) for word in label_array]

import umap

umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(vec_array)

kmeans_model = KMeans(n_clusters=100, init='k-means++', max_iter=1000)
kmeans_model.fit(umap_data, sample_weight=freq_weight)
y_pred = kmeans_model.fit_predict(umap_data, sample_weight=freq_weight)

fig = plt.figure()
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=y_pred)
centroidpoint = kmeans_model.cluster_centers_
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()

metrics.silhouette_score(umap_data, kmeans_model.labels_)
# km10:  0.31892186
# km100: 0.33703655

metrics.calinski_harabasz_score(umap_data, kmeans_model.labels_)
# km10:  4993.275308458779
# km100: 5492.335856872896

cen_id = []
cen_word = []
for i in range(10):
    c = kmeans_model.transform(umap_data)[:, i]
    id = np.where(c == c.min())[0][0]
    cen_id.append(id)
    cen_word.append(label_array[id])
# center words n_clusters = 10
# Out[115]:
# ['beyond',
#  'axonal',
#  'resistivity',
#  'collapse',
#  'integral',
#  'domesticated',
#  'glucagon',
#  'superconductor',
#  'coordinated',
#  'representation']


# center word, n_clusters = 100
# ['dilute',
#  'shadow',
#  'monolayers',
#  'mutation',
#  'pancreatic',
#  'allylic',
#  'around',
#  'computational',
#  'ubiquitination',
#  'identification',
#  'humans',
#  'generic',
#  'whose',
#  'complex',
#  'abscisic',
#  'arrest',
#  'evaporation',
#  'adrenoceptor',
#  'systems',
#  'subarctic',
#  'pressure',
#  'agents',
#  'dysplasia',
#  'south',
#  'thirst',
#  'significant',
#  'opposite',
#  'neurones',
#  'intron',
#  'oligonucleotides',
#  'waves',
#  'tension',
#  'hydrocarbons',
#  'protein',
#  'shift',
#  'grains',
#  'trace',
#  'alkali',
#  'suspensions',
#  'anthropogenic',
#  'improve',
#  'bumblebees',
#  'pigeons',
#  'affinities',
#  'pituitary',
#  'shells',
#  'reversals',
#  'social',
#  'metabotropic',
#  'calcium',
#  'nidulans',
#  'antipsychotic',
#  'luminescence',
#  'electric',
#  'functional',
#  'scallop',
#  'apolipoprotein',
#  'workers',
#  'normal',
#  'coincidence',
#  'erythrocyte',
#  'motor',
#  'microwave',
#  'lithosphere',
#  'bacteriorhodopsin',
#  'freely',
#  'similar',
#  'emission',
#  'atmospheric',
#  'paternal',
#  'sites',
#  'leukocyte',
#  'spliceosomal',
#  'epitope',
#  'geodetic',
#  'reconstruction',
#  'microcavity',
#  'stimulus',
#  'ratios',
#  'mechanisms',
#  'perforin',
#  'yoelii',
#  'potent',
#  'mother',
#  'bilaterian',
#  'billion',
#  'transfers',
#  'synergy',
#  'ischaemia',
#  'macromolecular',
#  'hydrated',
#  'allocation',
#  'gastrulation',
#  'celestial',
#  'contractility',
#  'sea',
#  'drinking',
#  'follicular',
#  'polarization',
#  'vaccinia']
