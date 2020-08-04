from util import *
from gensim.models.doc2vec import Doc2Vec
import gensim
import multiprocessing
from sklearn import utils
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

nat_meta = read_processed_meta('data/nat_meta.hdf5')
nat_meta = [nat_meta[i] for i in range(len(nat_meta)) if len(nat_meta[i].abstract) > 1 and len(nat_meta[i].title) > 0]
sci_meta = read_processed_meta('data/sci_meta.hdf5')
sci_meta = [sci_meta[i] for i in range(len(sci_meta)) if len(sci_meta[i].abstract) > 1 and len(sci_meta[i].title) > 0]


# Meta data of papers
class PaperMeta(object):
    def __init__(self, title, abstract, keyword, rating, url,
                 withdrawn, desk_reject, decision, author,  # review=None,
                 review_len=None, meta_review_len=0):
        self.title = title  # str
        self.abstract = abstract  # str
        self.keyword = keyword  # list[str]
        self.rating = rating  # list[int]
        self.url = url  # str
        self.withdrawn = withdrawn  # bool
        self.desk_reject = desk_reject  # bool
        self.decision = decision  # str
        # self.review = review           # list[str]
        self.author = author  # list[str]
        self.review_len = review_len  # list[int]
        self.meta_review_len = meta_review_len  # int
        if review_len is None or len(review_len) == 0:
            self.review_len_max = None
            self.review_len_min = None
        else:
            self.review_len_max = np.max(review_len)
            self.review_len_min = np.min(review_len)

        if len(self.rating) > 0:
            self.average_rating = np.mean(rating)
        else:
            self.average_rating = -1


def read_meta(filename):
    f = h5py.File(filename, 'r')
    meta_list = []
    for k in list(f.keys()):
        meta_list.append(PaperMeta(
            f[k]['title'].value,
            f[k]['abstract'].value,
            f[k]['keyword'].value.split('#'),
            f[k]['rating'].value,
            f[k]['url'].value,
            f[k]['withdrawn'].value,
            f[k]['desk_reject'].value,
            f[k]['decision'].value,
            # f[k]['review'].value if 'review' in list(f[k].keys()) else None,
            f[k]['author'].value.split('#') if 'author' in list(f[k].keys()) else None,
            f[k]['review_len'].value if 'review_len' in list(f[k].keys()) else None,
            f[k]['meta_review_len'].value if 'meta_review_len' in list(f[k].keys()) else None,
        ))
    return meta_list


stop_words = read_set_from_file('data/stopwords.txt')
neg_meta = read_meta('data/neg_meta.hdf5')
for m in neg_meta:
    m.title = process_string(m.title)
    m.abstract = process_string(m.abstract)
    m.title = [m.title[i] for i in range(len(m.title)) if m.title[i] not in stop_words]
    m.abstract = [m.abstract[i] for i in range(len(m.abstract)) if m.abstract[i] not in stop_words]

# tags_index = {'science': 0, 'nature': 1}
LabeledSentence = gensim.models.doc2vec.TaggedDocument
# for i in range(8000):
#     if i < 6000:
#         train_docs.append(LabeledSentence(words=(nat_meta[i].title + nat_meta[i].abstract), tags=[1]))
#         train_docs.append(LabeledSentence(words=(sci_meta[i].title + sci_meta[i].abstract), tags=[0]))
#     elif i >= 6000 and i < 8000:
#         test_docs.append(LabeledSentence(words=(nat_meta[i].title + nat_meta[i].abstract), tags=[1]))
#         test_docs.append(LabeledSentence(words=(sci_meta[i].title + sci_meta[i].abstract), tags=[0]))
#     else:
#         break

# if classify nat/sci from other docs, all docs' tags=[1], others' tags=[0]

train_docs = []
test_docs = []

for i in range(2000):
    if i < 1500:
        train_docs.append(LabeledSentence(words=(nat_meta[i].title + nat_meta[i].abstract), tags=[1]))
        train_docs.append(LabeledSentence(words=(sci_meta[i].title + sci_meta[i].abstract), tags=[1]))
    elif 1500 <= i < 2000:
        test_docs.append(LabeledSentence(words=(nat_meta[i].title + nat_meta[i].abstract), tags=[1]))
        test_docs.append(LabeledSentence(words=(sci_meta[i].title + sci_meta[i].abstract), tags=[1]))
    else:
        break
for i in range(2000):
    if i < 1000:
        train_docs.append(LabeledSentence(words=(neg_meta[i].title + neg_meta[i].abstract), tags=[0]))
    elif 1000 <= i < 2000:
        test_docs.append(LabeledSentence(words=(neg_meta[i].title + neg_meta[i].abstract), tags=[0]))

train_docs = utils.shuffle(train_docs)
test_docs = utils.shuffle(test_docs)
cores = multiprocessing.cpu_count()
d2v_model = Doc2Vec(dm=1, window=10, vector_size=100, min_count=2, workers=cores, alpha=0.025, min_alpha=0.001)
d2v_model.build_vocab([x for x in tqdm(train_docs)])
d2v_model.train(train_docs, total_examples=len(train_docs), epochs=30)

d2v_model.save('data/ns_other_classifier_doc2vec_model')


def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors


y_train, X_train = vector_for_learning(d2v_model, train_docs)
y_test, X_test = vector_for_learning(d2v_model, test_docs)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
# print('Testing accuracy for Nature and Science classifier %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score for Nature and Science classifier %s' % f1_score(y_test, y_pred, average='weighted'))

print('Testing accuracy for Nature, Science and other papers classifier %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score for Nature, Science and other papers classifier %s' % f1_score(y_test, y_pred,
                                                                                       average='weighted'))

# Testing accuracy for Nature and Science classifier 0.503
# Testing F1 score for Nature and Science classifier 0.502770155504382

# Testing accuracy for Nature, Science and other papers classifier 0.9515
# Testing F1 score for Nature, Science and other papers classifier 0.9513903244194715
