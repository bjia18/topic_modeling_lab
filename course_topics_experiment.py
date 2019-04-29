from preprocess_file import *
from print_model import *

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
    courses_stopwords=['course','student','students','work','prerequisite','level','instructor','class','study',
            'introduction','semester','process','various','also','well','permission','fee','rule','rules','sophomore',
            'place','process','various','break','women','texts','good','true','th','thea','theatre','designer',
            'options','u','specifically','fourth','topics','st','john','include','democratic','photographic','use',
            'especially','must','much','remarkable','new','many','sart','enrolled','register','du','cultural',
            'viewpoints','cows','written','already','announced','approved','optics','engines','prerequisites',
            'historical','economics','including']
    stopword_nltk.extend(courses_stopwords)
    #print(stopword_nltk)
    tokenized_list = [preprocess(doc, stopword_nltk) for doc in documents]

    # Create the Corpus
    mydict = corpora.Dictionary()
    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
    # pprint(mycorpus)
    # > [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], [(4, 4)]]

    word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]
    pprint(word_counts)

    # Create the TF-IDF model
    tfidf = models.TfidfModel(mycorpus, smartirs='Ltn')

    # Show the TF-IDF weights
    #for doc in tfidf[mycorpus]:
    #    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    # can also use corpus=tfidf[mycorpus]
    '''lda_model = LdaModel(corpus=tfidf[mycorpus],
                         id2word=mydict,
                         random_state=100,
                         num_topics=5,
                         passes=200,
                         chunksize=1000,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)'''
   
    my_stop_words = text.ENGLISH_STOP_WORDS.union(courses_stopwords)
    vectorizer = CountVectorizer(stop_words=my_stop_words)
    matrix = vectorizer.fit_transform(documents)
    
    guided_model=guidedlda.GuidedLDA(n_topics=5, n_iter=200, random_state=100, refresh=100)
    word2id = dict((v, idx) for idx, v in enumerate(vectorizer.get_feature_names()))
    seed_topic_list = [['science','mathematics','laboratory'],
                   ['production','theater','sound','film','video'],
                   ['literature', 'writing','read','novels'],
                   ['history','economic'],
                   ['gender','modern','american','world','social']]

    seed_topics = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            seed_topics[word2id[word]] = t_id
    guided_model.fit(matrix, seed_topics=seed_topics, seed_confidence=075)

    # save the model
    #guided_model.save('lda_model_courses.model')
    #corpora.MmCorpus.serialize('mycorpus.mm', tfidf[mycorpus])
    # See the topics
    #lda_model.print_topics(-1)
    #groups=lda_model.show_topics(-1)
    '''for i in groups:
        print(i)
        print()'''
    model=guided_model
    n_top_words = 10
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vectorizer.get_feature_names())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words))+'\n')
    
    with open('guidedlda_model.pickle', 'wb') as file_handle:
        pickle.dump(model, file_handle)
    '''courses=model
    corpus = tfidf[mycorpus]
    dominant_topics, topic_percentages = count_topics(courses, corpus=corpus, end=-1)
    topic_graph(dominant_topics, topic_percentages, courses)'''

    #lda_model.show_topic(0)

if __name__ == "__main__":
    main()
