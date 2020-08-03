import pyLDAvis
import pyLDAvis.gensim
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from util import *
import seaborn as sns
import matplotlib.pyplot as plt

stop_words = read_set_from_file('data/stopwords.txt')
nat_data = get_nature_data()
# title

title = nat_data.title.to_list()
texts = [[word for word in document.lower().split() if word not in stop_words] for document in title]

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
# vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
# pyLDAvis.display(vis_data)
# pyLDAvis.save_html(vis_data, 'nat_title_lda_visualization.html')

# fig = plt.figure(figsize=(15, 30))
# for i in range(5):
#     df = pd.DataFrame(lda.show_topic(i), columns=['term', 'prob']).set_index('term')  # print word-topic probablity
#     # df = df.sort_values('prob')
#     plt.subplot(5, 1, i + 1)
#     plt.title('topic ' + str(i))
#     sns.barplot(x='prob', y=df.index, data=df, label='Scholar', palette='Reds_d')
#     plt.xlabel('probability')
#
# plt.show()
# fig.savefig('nat_title_word_importance.png')

# p(word w with topic t) = p(topic t | document d) * p(word w | topic t)
# On the left,
# the topics are plotted as circles,
# whose centers are defined by the computed distance between topics (projected into 2 dimensions).
# The prevalence of each topic is indicated by the circle’s area.

# doc_topics = lda.get_document_topics(corpus, 0)
# probs = [[entry[1] for entry in doc] for doc in doc_topics]
# docs_topics_prob = np.array(probs)
#
#
def get_doc_topic_id(doc_id, docs_topics_prob):
    result = np.where(docs_topics_prob[doc_id] == docs_topics_prob[doc_id].max())
    return result[0][0]


#
#
# topic_doc_num = [0, 0, 0, 0, 0]
# for i in range(len(docs_topics_prob)):
#     topic_doc_num[get_doc_topic_id(i)] += 1
#
# fig = plt.figure(figsize=(15, 30))
# for i in range(5):
#     df = pd.DataFrame(lda.show_topic(i, 50), columns=['term', 'prob'])
#     df['freq'] = np.zeros([50, 1])
#     terms = df.term
#     for t in terms:
#         df.freq[df.term == t] = sum([1 if texts[j].count(t)>=1 else 0 for j in range(len(texts)) if get_doc_topic_id(j) == 0])
#     plt.subplot(5, 1, i + 1)
#     plt.title('topic ' + str(i))
#     df.freq /= topic_doc_num[i]
#     df = df.sort_values('freq', ascending=False)
#     df = df[:10]
#     sns.barplot(x='freq', y=df.term, data=df, label='Scholar', palette='Reds_d')
#     plt.xlabel('topic word in doc probability')
# plt.show()
# fig.savefig('asset/nature/nat_abstract_word_in_doc_prob.png')
#
# loading = LdaModel.load('data/nat_title_topic.model')
# loading.minimum_probability = 0
# abstract = nat_data.abstract.to_list()
# abstract_texts = [[word for word in document.lower().split() if word not in stop_words] for document in abstract]
#
# abstract_topics = [loading[dictionary.doc2bow(abstract_texts[i])] for i in range(len(abstract_texts)) if
#                    len(abstract_texts[i]) > 0]
# probs = [[entry[1] for entry in doc] for doc in abstract_topics]
# abs_topics_prob = np.array(probs)
#
# title_topics = lda.get_document_topics(corpus, 0)
# probs = [[entry[1] for entry in title_topics[i]] for i in range(len(title_topics)) if len(abstract_texts[i]) > 0]
# title_topics_prob = np.array(probs)
#
# count = 0
# for i in range(len(abs_topics_prob)):
#     if get_doc_topic_id(i, abs_topics_prob) == get_doc_topic_id(i, title_topics_prob):
#         count += 1
# count = 27514
# count / len(abs_topics_prob) = 0.5229406621811685
