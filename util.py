import numpy as np
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

from pandas import to_datetime


class PaperMeta(object):
    def __init__(self, title, date, author, abstract, affiliation, doi, citation):
        self.title = title
        self.date = date
        self.author = author  # series
        self.abstract = abstract
        self.affiliation = affiliation
        self.doi = doi
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
            if isinstance(data.citation[i], str):
                citation = data.citation[i].replace("['", '').replace("']", '').split("', '")
            else:
                citation = None

            meta_list.append(PaperMeta(
                title, date, author, abstract, affiliation, doi, citation
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

    return meta_list


def get_content_corpus(meta_list):
    content_corpus = []  # title + abstract + citation
    for m in meta_list:
        title = m.title
        abstract = m.abstract
        citation = m.citation
        content_corpus.extend(title)
        if abstract is not None:
            content_corpus.extend(abstract[j])
        if citation is not None:
            for j in citation:
                content_corpus.extend(citation[j])
    return content_corpus


def get_author_corpus(meta_list):
    author_corpus = []  # author
    for m in meta_list:
        author_corpus.extend(m.author)
    return author_corpus


def get_affiliation_corpus(meta_list):
    affiliation_corpus = []  # affiliation
    for m in meta_list:
        affiliation = m.affiliation
        if affiliation is not None:
            for j in affiliation:
                affiliation_corpus.extend(affiliation[j])
    return affiliation_corpus

