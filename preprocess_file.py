import gensim
from gensim import corpora
from pprint import pprint
from gensim import models
import numpy as np
from gensim.models import LdaModel, LdaMulticore
from gensim.utils import simple_preprocess, lemmatize
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import logging
import csv

def preprocess(doc, stop_words):
    words_processed = []

    doc_words = doc.split()
    for wd in doc_words:
        if wd not in stop_words:  # remove stopwords
            lemmatized_word = lemmatize(wd, allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
            if lemmatized_word:
                words_processed.append( lemmatized_word[0].split(b'/')[0].decode('utf-8'))
            words_processed.append(wd)
    return words_processed

def read_course_descriptions():
    columns=True

    doc_list = []
    with open('course-descriptions2.csv', errors='replace') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            if columns==True:
                columns=False
                continue
            text = row[2]
            clean_chars = [c.lower() if c.isalpha() else ' ' for c in text]
            text = "".join(clean_chars)
            doc_list.append(text)

    return doc_list
