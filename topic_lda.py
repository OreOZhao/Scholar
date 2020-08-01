import pyLDAvis
import pyLDAvis.gensim
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from util import *
import seaborn as sns
import matplotlib.pyplot as plt

stop_words = read_set_from_file('stopwords.txt')

# title
nat_data = get_nature_data()
title = nat_data.title.to_list()
texts = [[word for word in document.lower().split() if word not in stop_words] for document in title]
# texts = list(list(title words))

# # abstract
# abstract = nat_data.abstract.to_list()
# while None in abstract:
#     abstract.remove(None)
# texts = [[word for word in document.lower().split() if word not in stop_words] for document in abstract]

dictionary = corpora.Dictionary(texts)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
corpus = [dictionary.doc2bow(text) for text in texts]  # construct Bag Of Word corpus
tfidf_model = models.TfidfModel(corpus)  # construct tf-idf model
corpus_tfidf = tfidf_model[corpus]  # get corpus's tf-idf
# construct LDA model
lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=5)
lda.show_topics(5)
# prob is the distribution of LDA topic probability
# p(word w| topic t): probability of each word belonging to each topic
# p(topic t| document d): each article belongs to each topic by percentage of similarity
# data visualization
vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
# pyLDAvis.display(vis_data)
pyLDAvis.save_html(vis_data, 'nat_title_lda_visualization.html')

fig = plt.figure(figsize=(15, 30))
for i in range(5):
    df = pd.DataFrame(lda.show_topic(i), columns=['term', 'prob']).set_index('term')
    # df = df.sort_values('prob')
    plt.subplot(5, 1, i + 1)
    plt.title('topic ' + str(i))
    sns.barplot(x='prob', y=df.index, data=df, label='Scholar', palette='Reds_d')
    plt.xlabel('probability')

plt.show()
fig.savefig('nat_title_word_importance.png')

# p(word w with topic t) = p(topic t | document d) * p(word w | topic t)
# On the left,
# the topics are plotted as circles,
# whose centers are defined by the computed distance between topics (projected into 2 dimensions).
# The prevalence of each topic is indicated by the circle’s area.
