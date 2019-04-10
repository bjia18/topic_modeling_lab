# https://www.machinelearningplus.com/nlp/gensim-tutorial/
import gensim
from gensim import corpora
from pprint import pprint
from gensim import models
import numpy as np
from gensim.models import LdaModel, LdaMulticore
from gensim.utils import simple_preprocess, lemmatize
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import logging
import csv

def preprocess(doc, stop_words):
    words_processed = []

    doc_words = doc.split()
    for wd in doc_words:
        if wd not in stop_words:  # remove stopwords
            #lemmatized_word = lemmatize(wd, allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
            #if lemmatized_word:
            #words_processed.append( lemmatized_word[0].split(b'/')[0].decode('utf-8'))
            words_processed.append(wd)
    return words_processed


def main():
    # How to create a dictionary from a list of sentences?
    documents = read_course_descriptions()

    # Tokenize(split) the sentences into words
    texts = [[text for text in doc.split()] for doc in documents]

    # Create dictionary
    dictionary = corpora.Dictionary(texts)

    # Get information about the dictionary
    print(dictionary)

    # print(dictionary.token2id)

    # Tokenize the docs
    stopword_nltk = stopwords.words('english')
    tokenized_list = [preprocess(doc, stopword_nltk) for doc in documents]

    # Create the Corpus
    mydict = corpora.Dictionary()
    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
    # pprint(mycorpus)
    # > [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], [(4, 4)]]

    word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]
    pprint(word_counts)

    # Create the TF-IDF model
    tfidf = models.TfidfModel(mycorpus, smartirs='ntc')

    # Show the TF-IDF weights
    #for doc in tfidf[mycorpus]:
    #    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    lda_model = LdaModel(corpus=tfidf[mycorpus],
                         id2word=mydict,
                         random_state=100,
                         num_topics=30,
                         passes=100,
                         chunksize=1000,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=1000,
                         gamma_threshold=0.001,
                         per_word_topics=True)

    # save the model
    lda_model.save('lda_model.model')

    # See the topics
    #lda_model.print_topics(-1)

    lda_model.show_topic(0)

if __name__ == "__main__":
    main()