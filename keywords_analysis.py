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

corpus = get_corpus_from_file('content_corpus.txt')