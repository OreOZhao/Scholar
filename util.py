import numpy as np
import numpy.matlib
import string
import h5py
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
import glob
import math
import random
from pandas import to_datetime
import datetime


class PaperMeta(object):
    def __init__(self, title, date, author, abstract, affiliation, doi, citation, keywords=None, year=None):
        self.title = title
        self.date = date
        self.author = author  # series
        self.abstract = abstract
        self.affiliation = affiliation
        self.doi = doi
        self.keywords = keywords
        self.year = year
        if isinstance(citation, str):
            self.citation = citation
        else:
            self.citation = None


def write_meta(meta_list, filename):
    f = h5py.File(filename, 'w')
    for i, m in enumerate(meta_list):
        grp = f.create_group(str(i))
        grp['title'] = m.title
        grp['date'] = m.date.strftime('%Y%m%d')
        grp['author'] = '(split)'.join(m.author)
        if m.abstract is None:
            grp['abstract'] = '(none)'
        else:
            grp['abstract'] = m.abstract
        grp['affiliation'] = '(split)'.join(m.affiliation)
        grp['doi'] = m.doi
        if m.citation is None:
            grp['citation'] = '(none)'
        else:
            grp['citation'] = '(split)'.join(m.citation)
    f.close()


def read_meta(filename):
    f = h5py.File(filename, 'r')
    meta_list = []
    for k in list(f.keys()):
        meta_list.append(PaperMeta(
            f[k]['title'].value,
            f[k]['date'].value,
            f[k]['author'].value.split('(split)'),
            f[k]['abstract'].value,
            f[k]['affiliation'].value.split('(split)'),
            f[k]['doi'].value,
            f[k]['citation'].value.split('(split)')
        ))
    return meta_list


def get_data():
    db = pymysql.connect(host="localhost", user="root", password="root", db="scholar")
    cursor = db.cursor()
    engine = create_engine("mysql+pymysql://root:root@localhost/scholar")
    # sql = "select * from %s"
    data_sci = pd.io.sql.read_sql_table("science", engine)
    data_sci.date = pd.to_datetime(data_sci.date, format="%Y-%m-%d")
    # print(type(data_sci), '\n', data_sci)
    data_nat = pd.io.sql.read_sql_table("nature", engine)
    data_nat.date = pd.to_datetime(data_nat.date, format="%d-%m-%Y")

    # print(type(data_nat), '\n', data_nat)
    data = pd.concat([data_nat, data_sci], axis=0, ignore_index=True)
    for i in data.index:
        if data.title[i] is None:
            data = data.drop(index=i)
    return data


def get_science_data():
    db = pymysql.connect(host="localhost", user="root", password="root", db="scholar")
    cursor = db.cursor()
    engine = create_engine("mysql+pymysql://root:root@localhost/scholar")
    # sql = "select * from %s"
    data_sci = pd.io.sql.read_sql_table("science", engine)
    data_sci.date = pd.to_datetime(data_sci.date, format="%Y-%m-%d")
    # print(type(data_sci), '\n', data_sci)
    # print(type(data_nat), '\n', data_nat)
    data = data_sci
    for i in data.index:
        if data.title[i] is None:
            data = data.drop(index=i)
    return data


def get_nature_data():
    db = pymysql.connect(host="localhost", user="root", password="root", db="scholar")
    cursor = db.cursor()
    engine = create_engine("mysql+pymysql://root:root@localhost/scholar")
    # sql = "select * from %s"
    # print(type(data_sci), '\n', data_sci)
    data_nat = pd.io.sql.read_sql_table("nature", engine)
    data_nat.date = pd.to_datetime(data_nat.date, format="%d-%m-%Y")

    # print(type(data_nat), '\n', data_nat)
    data = data_nat
    for i in data.index:
        if data.title[i] is None:
            data = data.drop(index=i)
    return data


def crawl_meta(meta_hdf5=None, write_meta_name='data.hdf5'):
    if meta_hdf5 is None:
        meta_list = []
        data = get_data()
        for i in data.index:
            title = data.title[i]
            date = data.date[i]
            author = data.author[i].replace('\\', '').replace("['", '').replace("']", '').split("', '")
            abstract = data.abstract[i]
            affiliation = data.affiliation[i].replace("['", '').replace("']", '').split("', '")
            doi = data.doi[i]
            if 'citation' in data.keys() and isinstance(data.citation[i], str):
                citation = data.citation[i].replace("['", '').replace("']", '').split("', '")
            else:
                citation = None

            meta_list.append(PaperMeta(
                title, date, author, abstract, affiliation, doi, citation, year=None, keywords=None
            ))

        write_meta(meta_list, write_meta_name)
    else:
        meta_list = read_meta(meta_hdf5)
    return meta_list


def process_string(s):
    s = s.lower()
    s = re.sub('<[^<>]+>', ' ', s)  # sub all the html
    s = re.sub('(http|https)://[^\s]*', 'httpaddr', s)
    s = re.sub('\d+', 'number', s)

    stemmer = nltk.stem.PorterStemmer()
    tokens = re.split('[ @$/#.-:&*+=\[\]?!()\{\},\'\">_<;%]', s)
    tokenlist = []
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        try:
            token = stemmer.stem(token)
        except:
            token = ''
        if len(token) < 1:
            continue  # 字符串长度小于1的不添加到tokenlist里
        tokenlist.append(token)

    return tokenlist


def process_list(str_list):
    prolist = []
    for i in str_list:
        tokenlist = process_string(i)
        prolist.append(tokenlist)
    return prolist


def process_abstract(abstract):
    if abstract is not None:
        return process_string(abstract)
    else:
        return None


def process_author(author):
    if isinstance(author, list):
        return process_list(author)
    else:
        return None


def process_affiliation(affiliation):
    if isinstance(affiliation, list):
        return process_list(affiliation)
    else:
        return None


def process_citation(citation):
    if isinstance(citation, list):
        return process_list(citation)
    else:
        return None


def process_meta_list(meta_list):
    for m in meta_list:
        m.title = process_string(m.title)
        m.abstract = process_abstract(m.abstract)
        m.citation = process_citation(m.citation)
        m.affiliation = process_affiliation(m.affiliation)
        m.date = datetime.datetime.strptime(m.date, "%Y%m%d")
        m.year = m.date.year
    process_keywords(meta_list)
    return meta_list


def process_keywords(meta_list):
    stop_words = read_set_from_file('stopwords.txt')
    for m in meta_list:
        keywords = []
        for word in m.title:
            if word not in stop_words:
                keywords.append(word)
        m.keywords = keywords


def get_title_corpus(meta_list, filename):
    corpus = []  # title
    for m in meta_list:
        title = m.title
        corpus.extend(title)
    write_corpus(corpus, filename)
    return corpus


def get_abstract_corpus(meta_list, filename):
    corpus = []  # title
    for m in meta_list:
        abstract = m.abstract
        if abstract is not None:
            corpus.extend(abstract)
    write_corpus(corpus, filename)
    return corpus


def get_citation_corpus(meta_list, filename):
    corpus = []  # title
    for m in meta_list:
        citation = m.citation
        if citation is not None:
            for j in citation:
                corpus.extend(j)
    write_corpus(corpus, filename)
    return corpus


def get_content_corpus(meta_list):
    content_corpus = []  # title + abstract + citation
    for m in meta_list:
        title = m.title
        abstract = m.abstract
        citation = m.citation
        content_corpus.extend(title)
        if abstract is not None:
            content_corpus.extend(abstract)
        if citation is not None:
            for j in citation:
                content_corpus.extend(j)
    write_corpus(content_corpus, 'content_corpus.txt')
    return content_corpus


def get_author_corpus(meta_list, filename):
    author_corpus = []  # author
    for m in meta_list:
        author_corpus.extend(m.author)
    write_corpus(author_corpus, filename)
    return author_corpus


def get_affiliation_corpus(meta_list):
    affiliation_corpus = []  # affiliation
    for m in meta_list:
        affiliation = m.affiliation
        if affiliation is not None:
            for j in affiliation:
                affiliation_corpus.extend(j)
    write_corpus(affiliation_corpus, 'affiliation_corpus.txt')
    return affiliation_corpus


def write_corpus(corpus, filename):
    f = open(filename, 'w')
    for i in range(len(corpus)):
        f.write('%s\n' % corpus[i])
    f.close()


def get_corpus_from_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    corpus = []
    for i in lines:
        corpus.append(i.replace('\n', ''))
    return corpus


def content_indices(m, content_corpus):
    indices = []  # title + abstract + citation
    title = m.title
    abstract = m.abstract
    citation = m.citation
    for j in range(len(content_corpus)):
        for word in title:
            if word == content_corpus[j]:
                indices.append(j)
        for word in abstract:
            if word == content_corpus[j]:
                indices.append(j)
        if citation is not None:
            for word in citation:
                if word == content_corpus[j]:
                    indices.append(j)
    return indices


def author_indices(m, author_corpus):
    indices = []
    for j in range(len(author_corpus)):
        author = m.author
        for a in author:
            if a == author_corpus[j]:
                indices.append(j)
    return indices


def affiliation_indices(m, affiliation_corpus):
    indices = []
    for j in range(len(affiliation_corpus)):
        affiliation = m.affiliation
        for a in affiliation:
            if a == affiliation_corpus[j]:
                indices.append(j)
    return indices


def get_feature(meta_list):
    content_corpus = get_corpus_from_file('most2000_content.txt')
    author_corpus = get_corpus_from_file('most2000_author.txt')
    affiliation_corpus = get_corpus_from_file('most2000_affiliation.txt')
    feature_len = len(content_corpus) + len(author_corpus) + len(affiliation_corpus)
    matrix = np.zeros(feature_len)
    sample = random.sample(meta_list, 4000)
    for m in sample:
        feature = np.zeros(feature_len, dtype=int)
        co_indices = content_indices(m, content_corpus)
        for each in co_indices:
            feature[each] = 1
        au_indices = author_indices(m, author_corpus)
        for each in au_indices:
            feature[each + len(content_corpus)] = 1
        af_indices = affiliation_indices(m, affiliation_corpus)
        for each in af_indices:
            feature[each + len(content_corpus) + len(author_corpus)] = 1
        matrix = np.vstack((matrix, feature))
    np.delete(matrix, 0, axis=0)
    y = np.ones((len(matrix), 1), dtype=int)
    scio.savemat('train_data.mat', {'X': matrix, 'y': y})


def read_set_from_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    read_set = []
    for line in lines:
        read_set.append(line.replace('\n', ''))
    f.close()
    return set(read_set)


def write_set_to_file(filename, write_set):
    f = open(filename, 'w')
    for word in write_set:
        f.write('%s\n' % word)
    f.close()


def get_keywords_dict_by_year(meta_list):
    dict = {}
    for m in meta_list:
        if m.year in dict:
            for k in m.keywords:
                if k in dict[m.year]:
                    dict[m.year][k] += 1
                else:
                    dict[m.year][k] = 1
        else:
            dict[m.year] = {}
    return dict


def get_subdict_keywords(dict, threshold):
    subdict = {}
    for key, value in dict.items():
        subdict[key] = {}
    for key, value in dict.items():
        for k, v in value.items():
            if v > threshold:
                subdict[key][k] = v
    return subdict


def write_processed_meta(meta_list, filename):
    f = h5py.File(filename, 'w')
    for i, m in enumerate(meta_list):
        grp = f.create_group(str(i))
        grp['title'] = '(split)'.join(m.title)
        grp['date'] = m.date.strftime('%Y%m%d')
        grp['author'] = '(split)'.join(m.author)
        if m.abstract is None:
            grp['abstract'] = '(none)'
        else:
            grp['abstract'] = '(split)'.join(m.abstract)
        temp_aff = []
        for a in m.affiliation:
            temp = ''
            temp += '(split)'.join(a)
            temp_aff.append(temp)
        grp['affiliation'] = '(group)'.join(temp_aff)
        grp['doi'] = m.doi
        if m.citation is None:
            grp['citation'] = '(none)'
        else:
            grp['citation'] = '(split)'.join(m.citation)
        grp['keywords'] = '(split)'.join(m.keywords)
        grp['year'] = m.year
    f.close()


def read_processed_meta(filename):
    f = h5py.File(filename, 'r')
    meta_list = []
    for k in list(f.keys()):
        meta_list.append(PaperMeta(
            title=f[k]['title'].value.split('(split)'),
            date=datetime.datetime.strptime(f[k]['date'].value, "%Y%m%d"),
            author=f[k]['author'].value.split('(split)'),
            abstract=f[k]['abstract'].value.split('(split)'),
            affiliation=f[k]['affiliation'].value.split('(group)'),
            doi=f[k]['doi'].value,
            citation=f[k]['citation'].value.split('(split)'),
            keywords=f[k]['keywords'].value.split('(split)'),
            year=f[k]['year'].value
        ))
    f.close()
    for m in meta_list:
        for i in range(len(m.affiliation)):
            m.affiliation[i] = m.affiliation[i].split('(split)')
    return meta_list


def get_keywords_df(subdict):
    x_year = []
    y_freq = []
    keywords = []
    for key, value in subdict.items():
        for k, v in value.items():
            x_year.append(key)
            y_freq.append(v)
            keywords.append(k)
    df = pd.DataFrame({'Year': x_year, 'Frequency': y_freq, 'Keyword': keywords})
    df = df.sort_values(by='Year', ascending=True)
    return df


def tf_idf_keywords(dict):
    tf_idf = dict
    tf = {}
    for key in dict.keys():
        tf[key] = {}
    for key, value in dict.items():
        for k, v in dict[key].items():
            tf[key][k] = v / len(dict[key])

    idf = {}
    a = pd.DataFrame.from_dict(dict)
    for i in list(a.index):
        idf[i] = 0
    for i in idf.keys():
        for k in a.keys():
            if not math.isnan(a[k][i]):
                idf[i] += 1
    for k in idf.keys():
        idf[k] = math.log(51 / idf[k] + 1)
    for key, value in dict.items():
        for k, v in value.items():
            tf_idf[key][k] = tf[key][k] * idf[k]
    return tf_idf
