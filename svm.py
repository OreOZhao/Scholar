import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import svm
import re
import nltk
import pymysql
from sqlalchemy import create_engine
import string
import seaborn
import time
import os.path
import glob
from util import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

CRAWL_DATA = False

if os.path.exists('data.hdf5'):
    CRAWL_DATA = False

if CRAWL_DATA:
    meta_list = crawl_meta(
        meta_hdf5=None,
        write_meta_name='data.hdf5'
    )
else:
    meta_hdf5 = glob.glob('data.hdf5')[0]
    meta_list = crawl_meta(meta_hdf5=meta_hdf5)

print('Number of Articles: %d' % len(meta_list))

count_article = len(meta_list)

# data = get_data()


# freq = pd.Series(data.abstract[1].split()).value_counts()[:20]


# stop_words = set(stopwords.words("english"))

print('Processing Meta list')
meta_list = process_meta_list(meta_list)
# content_corpus = get_content_corpus(meta_list)
# author_corpus = get_author_corpus(meta_list)
# affiliation_corpus = get_affiliation_corpus(meta_list)
# content_corpus = get_corpus_from_file('content_corpus.txt')


# train = scio.loadmat('train.mat')
# train_x = train['x']
# train_y = train['y']
# clf = svm.SVC(C=0.1, kernel='linear')
# clf.fit(train_x, train_y)


def accuracy(clf, x, y):
    predict_y = clf.predict(x)
    m = y.size
    count = 0
    for i in range(m):
        count = count + np.abs(int(predict_y[i]) - int(y[i]))
    return 1 - float(count / m)
#
#
# accuracy(clf, train_x, train_y)

