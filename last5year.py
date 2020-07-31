from util import *
import pandas as pd

nat_meta = read_processed_meta('data/nat_meta.hdf5')
meta_list = [nat_meta[i] for i in range(len(nat_meta)) if nat_meta[i].year >= 2015]
#
# # authors
# d = {'author': [], 2015: [], 2016: [], 2017: [], 2018: [], 2019: [], 2020: []}
# df = pd.DataFrame(data=d)
#
# df.set_index('author')
# dd = {}
# for m in meta_list:
#     for a in m.author:
#         if a in dd.keys():
#             if m.year in dd[a].keys():
#                 dd[a][m.year] += 1
#             else:
#                 dd[a][m.year] = 1
#         else:
#             dd[a] = {}
#             dd[a][m.year] = 1
#
# df = pd.DataFrame(dd)
# df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
# df = df2.sort_index(axis=1)
# for i in range(len(df)):
#     for j in range(2015, 2020):
#         if math.isnan(df.iloc[i][j]):
#             df.iloc[i][j] = 0
#
# df = df.reset_index()
# df = df.rename(columns={'index': 'author'})

nat_authors = get_author_corpus(nat_meta, 'nat_author_corpus.txt')
nat_authors_recent = get_author_corpus(meta_list, 'nat_author_recent_corpus.txt')
temp = []
for a in nat_authors:
    if len(a) >= 5:
        temp.append(a)
nat_authors = temp
temp = []
for a in nat_authors_recent:
    if len(a) >= 5:
        temp.append(a)
nat_authors_recent = temp

temp = pd.Series(nat_authors).value_counts()
temp2 = pd.Series(pd.Series(nat_authors_recent).value_counts(), dtype=float)
temp3 = temp2
for i in temp2.index:
    temp3[i] = float(temp2[i] / temp[i])
temp2 = pd.Series(pd.Series(nat_authors_recent).value_counts(), dtype=float)
temp4 = [{i: temp2[i] for i in temp2.index if temp3[i] == 1}]
temp5 = pd.DataFrame(data=temp4)
df = pd.DataFrame(temp5.values.T, index=temp5.columns)[:50]
key = [index for index in df.index]
value = [int(value) for value in df.values]
y_pos = np.arange(len(key))
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(y_pos, value, align='center', color='green', ecolor='black', log=True)
ax.set_yticks(y_pos)
ax.set_yticklabels(key, rotation=0, fontsize=10)
ax.invert_yaxis()
for i, v in enumerate(value):
    ax.text(v + 3, i + .25, str(v), color='black', fontsize=10)

ax.set_xlabel('Number of Papers')
ax.set_ylabel('Author')
ax.set_title('Nature Top 50 Authors in Last 5 Years')
plt.show()
fig.savefig('asset/nat_top50authors_last5year.png')
