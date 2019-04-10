from model_topics import *
import os

def read_songs(main_dir):
    directory = os.fsencode(main_dir)

    doc_list = []
    for sub_dir in os.listdir(directory):
        country_dir = os.fsdecode(sub_dir)
        for filename in os.listdir(os.path.join(main_dir, country_dir)):
            f = open(os.path.join(main_dir,country_dir, filename))
            text = f.read()
            clean_chars = [c.lower() if c.isalpha() else ' ' for c in text]
            text = "".join(clean_chars)
            text_words = text.split()
            text = " ".join(text_words)
            doc_list.append(text)

    return doc_list


def main():
    # How to create a dictionary from a list of sentences?
    documents = read_songs("country_songs")

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
    # for doc in tfidf[mycorpus]:
    #    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    # can also use corpus=tfidf[mycorpus]
    lda_model = LdaModel(corpus=mycorpus,
                         id2word=mydict,
                         random_state=100,
                         num_topics=25,
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
    # lda_model.print_topics(-1)

    lda_model.show_topic(0)

if __name__ == "__main__":
    main()