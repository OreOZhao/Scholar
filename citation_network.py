from util import *

nat_data = get_nature_data()
nat_dict = {'title': nat_data.title, 'doi': nat_data.doi, 'citation': nat_data.citation, 'cite_doi': []}
df = pd.DataFrame(nat_dict)

for i in df.index:
    if 'doi' not in df.citation[i] and 'DOI' not in df.citation[i]:
        df = df.drop(index=i)

df = df.reset_index(drop=True)


def get_cite_doi(cite_list):
    cmpl = re.compile('(http|https)://[^\s]*')
    cite_doi = [i for c in cite_list for i in c.split() if cmpl.match(i)]
    return cite_doi


# remove citation not have doi
for i in df.index:
    c = df.citation[i].replace("['", '').replace("']", '').split("', '")
    temp = []
    for j in range(len(c)):
        if 'doi' in c[j] or 'DOI' in c[j]:
            temp.append(c[j])
    df.citation[i] = temp
    df.cite_doi[i] = get_cite_doi(temp)

for i in df.index:
    if len(df.cite_doi[i]) == 0:
        df = df.drop(index=i)
df = df.reset_index(drop=True)
df['year'] = ''
for i in df.index:
    df.year[i] = nat_data[nat_data.doi == df.doi[i]].year.values[0]

df.to_csv('nat_cite.csv')
df = pd.read_csv('nat_cite.csv', index_col=0)