from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path
from PIL import Image
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from util import *

stop_words = set(STOPWORDS)

corpus = open('content_corpus.txt', 'r').read()


co_wc = WordCloud(
    background_color='white',
    stopwords=stop_words,
    max_words=200,
    max_font_size=50,
    random_state=42
).generate_from_text(corpus)
print(co_wc)
fig = plt.figure(1)
plt.imshow(co_wc)
plt.axis('off')
plt.show()
fig.savefig('co_wc.png', dpi=900)
#
# au_wc = WordCloud(
#     background_color='white',
#     stopwords=stop_words,
#     max_words=200,
#     max_font_size=50,
#     random_state=42
# ).generate(str(author_corpus))
# print(au_wc)
# fig = plt.figure(1)
# plt.imshow(au_wc)
# plt.axis('off')
# plt.show()
# plt.savefig('au_wc.png', dpi=900)
#
# af_wc = WordCloud(
#     background_color='white',
#     stopwords=stop_words,
#     max_words=200,
#     max_font_size=50,
#     random_state=42
# ).generate(str(affiliation_corpus))
# print(af_wc)
# fig = plt.figure(1)
# plt.imshow(af_wc)
# plt.axis('off')
# plt.show()
# fig.savefig('af_wc.png', dpi=900)