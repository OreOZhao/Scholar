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
from util import get_data, crawl_meta
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# CRAWL_DATA = True
#
# if os.path.exists('data.hdf5'):
#     CRAWL_DATA = False
#
# if CRAWL_DATA:
#     meta_list = crawl_meta(
#         meta_hdf5=None,
#         write_meta_name='data.hdf5'
#     )
# else:
#     meta_hdf5 = glob.glob('data.hdf5')[0]
#     meta_list = crawl_meta(meta_hdf5=meta_hdf5)
#
# print('Number of Articles: %d' % len(meta_list))


data = get_data()


# freq = pd.Series(data.abstract[1].split()).value_counts()[:20]



# # stop_words = set(stopwords.words("english"))
# for i in data.index:
#     abstract = data.abstract[i]
#     if abstract is not None:
#         pro_abs = processString(abstract)
#     else:
#         pro_abs = None
