import gensim
from gensim import corpora
from pprint import pprint
from gensim import models
from gensim.models import LdaModel, LdaMulticore
from gensim.utils import simple_preprocess, lemmatize
import guidedlda
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from six.moves import cPickle as pickle
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt

def count_topics(model, corpus, start=0, end=1):
    # Sentence Coloring of N Sentences
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)

    return(dominant_topics, topic_percentages)

def topic_graph(dominant_topics, topic_percentages, model):
    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count')
    #df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
    index=df_dominant_topic_in_each_doc.index.tolist()
    for i in range(len(model.show_topics(-1))):
        if i not in index:
            df_dominant_topic_in_each_doc.loc[i] = pd.Series({'count':0})
    df_dominant_topic_in_each_doc.sort_index(inplace=True)
    df_dominant_topic_in_each_doc2=df_dominant_topic_in_each_doc.reset_index()
    
    # Total Topic Distribution by actual weight

    # Plot
    # Topic Distribution by Dominant Topics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120, sharey=True)
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc2, width=0.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc2.Dominant_Topic.unique().__len__()))
    
    #ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')
    ax1.set_ylim(0, 600)
    
    df_dominant_topic_in_each_doc.sort_values(['count'], ascending=False, inplace=True)
    #df_dominant_topic_in_each_doc=df_dominant_topic_in_each_doc.reset_index()
    ax2.bar(x=range(df_dominant_topic_in_each_doc2.Dominant_Topic.unique().__len__()), height='count', data=df_dominant_topic_in_each_doc, width=0.5, color='steelblue')
    ax2.set_xticks(range(df_dominant_topic_in_each_doc2.Dominant_Topic.unique().__len__()))
    ax2.set_xticklabels(df_dominant_topic_in_each_doc.index)

    #ax1.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Topics Sorted by Documents', fontdict=dict(size=10))
    ax2.set_ylabel('Number of Documents')
    ax2.set_ylim(0, 600)

    plt.savefig('topic_graph.png')
    plt.show()


def run():
    courses=LdaModel.load('lda_model_courses.model')
    #for i in courses.show_topics(formatted=False,num_topics=courses.num_topics,num_words=len(courses.id2word)):
    #    print (i)
    groups=courses.show_topics(-1)
    for i in groups:
        print(i)
        print()
    corpus = corpora.MmCorpus('mycorpus.mm')
    dominant_topics, topic_percentages = count_topics(courses, corpus=corpus, end=-1)
    topic_graph(dominant_topics, topic_percentages, courses)
