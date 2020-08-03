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

train_docs = []
test_docs = []
# tags_index = {'science': 0, 'nature': 1}
LabeledSentence = gensim.models.doc2vec.TaggedDocument
for i in range(8000):
    if i < 6000:
        train_docs.append(LabeledSentence(words=(nat_meta[i].title + nat_meta[i].abstract), tags=[1]))
        train_docs.append(LabeledSentence(words=(sci_meta[i].title + sci_meta[i].abstract), tags=[0]))
    elif i >= 6000 and i < 8000:
        test_docs.append(LabeledSentence(words=(nat_meta[i].title + nat_meta[i].abstract), tags=[1]))
        test_docs.append(LabeledSentence(words=(sci_meta[i].title + sci_meta[i].abstract), tags=[0]))
    else:
        break

# if classify nat/sci from other docs, all docs' tags=[1]

train_docs = utils.shuffle(train_docs)
test_docs = utils.shuffle(test_docs)
cores = multiprocessing.cpu_count()
d2v_model = Doc2Vec(dm=1, window=10, vector_size=500, min_count=2, workers=cores, alpha=0.025, min_alpha=0.001)
d2v_model.build_vocab([x for x in tqdm(train_docs)])
d2v_model.train(train_docs, total_examples=len(train_docs), epochs=30)


def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors


y_train, X_train = vector_for_learning(d2v_model, train_docs)
y_test, X_test = vector_for_learning(d2v_model, test_docs)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Testing accuracy for Nature and Science classifier %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score for Nature and Science classifier %s' % f1_score(y_test, y_pred, average='weighted'))
# Testing accuracy for Nature and Science classifier 0.503
# Testing F1 score for Nature and Science classifier 0.502770155504382
