import pandas as pd
import seaborn as sns
from util import *
import matplotlib.pyplot as plt
import glob

# meta_hdf5 = glob.glob('data.hdf5')[0]
# meta_list = crawl_meta(meta_hdf5=meta_hdf5)
#
# dict = get_keywords_dict_by_year(meta_list)
#
# subdict = get_subdict_keywords(dict)
#
#
# def get_keywords_df(subdict):
#     x_year = []
#     y_freq = []
#     keywords = []
#     for key, value in subdict.items():
#         for k, v in value.items():
#             x_year.append(key)
#             y_freq.append(v)
#             keywords.append(k)
#     df = pd.DataFrame({'Year': x_year, 'Frequency': y_freq, 'Keyword': keywords})
#     df = df.sort_values(by='Year', ascending=True)
#     return df
#
#
# df = get_keywords_df(subdict)
# sns.set(font_scale=1)
# fig = plt.figure(figsize=(30, 20))
# p1 = sns.regplot(data=df, x="Year", y="Frequency", fit_reg=False, marker="o", color="red", logx=False,
#                  scatter_kws={'s': 8})
# for line in range(0, df.shape[0]):
#     p1.text(df.Year[line], df.Frequency[line], df.Keyword[line], horizontalalignment='left', size='medium',
#             color='black', alpha=0.6)
# plt.show()
# fig.savefig('asset/year_freq.png')

stop_words = read_set_from_file('stopwords.txt')

corpus = get_corpus_from_file('content_corpus.txt')
temp = []
for word in corpus:
    if word not in stop_words:
        temp.append(word)

write_corpus(temp, 'content_without_stopwords.txt')  
temp = pd.DataFrame(temp)

temp.apply(pd.value_counts)[:50]
top50 = temp.apply(pd.value_counts)[:50]


key = [index for index in top50.index]
value = [int(value) for value in top50.values]
y_pos = np.arange(len(key))
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(y_pos, value, align='center', color='blue', ecolor='black', log=True)
ax.set_yticks(y_pos)
ax.set_yticklabels(key, rotation=0, fontsize=10)
ax.invert_yaxis()
for i, v in enumerate(value):
    ax.text(v + 3, i + .25, str(v), color='black', fontsize=10)

ax.set_xlabel('Frequency')
ax.set_title('Nature & Science Top 50 Keywords')
plt.show()
fig.savefig('asset/top50keywords.png')