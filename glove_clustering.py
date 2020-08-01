import codecs
import json
from numbers import Number

from sklearn.cluster import KMeans

from util import *

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

vector_file = '/Users/limingxia/downloads/glove.6b/glove.6B.50d.txt'
corpus = get_corpus_from_file('nat_title_origin_corpus.txt')


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
        for i, line in enumerate(f):
            sr = line.split()
            if sr[0] in corpus:
                labels_array.append(sr[0])
                vec_arrays.append(np.array([float(j) for j in sr[1:]]))
    return np.array(vec_arrays), labels_array


# affiliation clustering
class autovivify_list(dict):
    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


def find_word_clusters(labels_array, cluster_labels):
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words


def save_json(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f)


vec_arrays, labels_array = build_word_vector_matrix(corpus, vector_file)

kmeans_model = KMeans(init='k-means++', n_clusters=100, n_jobs=10, n_init=10)
vec_arrays = np.array(vec_arrays)
kmeans_model.fit(vec_arrays)
cluster_labels = kmeans_model.labels_
cluster_to_words = list(find_word_clusters(labels_array, cluster_labels).values())
save_json('nat_title_50D_100_clusters.json', cluster_to_words)
