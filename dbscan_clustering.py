from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from util import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from scipy.spatial.distance import cosine
from scipy.stats.stats import pearsonr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

scaler = StandardScaler()
abstracts = get_corpus_from_file('data/nat_abstract.txt')
while 'None' in abstracts:
    abstracts.remove('None')

tfidfvec = TfidfVectorizer(ngram_range=(1, 100), min_df=0.1, max_df=1.0, decode_error='ignore')
tfidfvec.fit(abstracts)
X = tfidfvec.fit_transform(abstracts).toarray()
db = DBSCAN(eps=0.3, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels1 = db.labels_
n_clusters_ = len(set(labels1)) - (1 if -1 in labels1 else 0)
clusters1 = {}
for c, i in enumerate(labels1):
    if i == -1:
        continue
    elif i in clusters1:
        clusters1[i].append(abstracts[c])
    else:
        clusters1[i] = [abstracts[c]]
for c in clusters1:
    print(clusters1[c])
    print()

unique_labels = set(labels1)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    class_member_mask = (labels1 == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markeredgecolor='k', markersize=14)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markeredgecolor='k', markersize=6)

plt.axis([-0.5, 1.5, -0.5, 1.5])
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
