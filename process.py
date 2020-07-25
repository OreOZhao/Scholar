from util import *
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from collections import Counter

def write_stopwords(sw):
    f = open('stopwords.txt', 'w')
    for w in sw:
        f.write('%s\n' % w)
    f.close()


corpus = get_corpus_from_file('content_corpus.txt')
stop_words = set(STOPWORDS)
sw = set(['via', 'use', 'number',  'make', 'among', 'year', 'per', 'will', 'mine', 'new',
           'howev', 'etc', 'thi', 'ha',  'two',  'wa',
           'may', 'offer', 'need', 'well', 'one', 'onto', 'object',
          'becaus', 'us', 'within', 'therebi', 'onli', 's', 'time'])
stop_words = stop_words.union(sw)
write_stopwords(stop_words)
temp = []
for word in corpus:
    if word not in stop_words:
        temp.append(word)
corpus = pd.Series(temp)
# freq = pd.Series(corpus).value_counts()[:20]
write_corpus(corpus, 'content_corpus.txt')

t = Counter(corpus)
most2000 = []
t = t.most_common(2000)
for i in range(len(t)):
    most2000.append(t[i][0])
write_corpus(most2000, 'most2000.txt')