from preprocess_file import *
from print_model import *
from class_clustering import read_file

from PIL import Image
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS, get_single_color_func
import matplotlib.colors as mcolors
import ast

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

def guided_topic_graph(topic_count, model):
    
    dominant_topics=[0,0,0,0,0]
    for i in topic_count:
        dominant_topics[i.index(max(i))]+=1
    
    df=pd.DataFrame({'count': dominant_topics})
    df2=df.reset_index()
    df.sort_values(['count'],inplace=True,ascending=False)
    
    # Total Topic Distribution by actual weight

    # Plot
    # Topic Distribution by Dominant Topics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120, sharey=True)
    ax1.bar(x='index', height='count', data=df2, width=0.5, color='firebrick')
    ax1.set_xticks(range(df2.index.unique().__len__()))

    #ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')
    ax1.set_ylim(0, 600)

    ax2.bar(x=range(df2.index.unique().__len__()), height='count', data=df, width=0.5, color='steelblue')
    ax2.set_xticks(range(df2.index.unique().__len__()))
    ax2.set_xticklabels(df.index)

    #ax1.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Topics Sorted by Documents', fontdict=dict(size=10))
    ax2.set_ylabel('Number of Documents')
    ax2.set_ylim(0, 600)

    plt.savefig('guided_topic_graph.png')
    plt.show()

def print_topics(model,printing):
    documents = read_course_descriptions()
    courses_stopwords=['course','student','students','work','prerequisite','level','instructor','class','study',
            'introduction','semester','process','various','also','well','permission','fee','rule','rules',
            'sophomore','place','process','various','break','women','texts','good','true','th','thea','theatre',
            'designer',
            'options','u','specifically','fourth','topics','st','john','include','democratic','photographic','use',
            'especially','must','much','remarkable','new','many','sart','enrolled','register','du','cultural',
            'viewpoints','cows','written','already','announced','approved','optics','engines','prerequisites',
            'historical','economics','including']

    my_stop_words = text.ENGLISH_STOP_WORDS.union(courses_stopwords)
    vectorizer = CountVectorizer(stop_words=my_stop_words)
    matrix = vectorizer.fit_transform(documents)

    n_top_words = 10
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vectorizer.get_feature_names())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        if printing==True:
            print('Topic {}: {}'.format(i, ' '.join(topic_words))+'\n')

    return (vectorizer.get_feature_names(), my_stop_words)

def word_clouds(model, vocab, stop_words, n_top_words, labeled):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=n_top_words,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    
    topics=[]
    topic_word=model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        words=np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1].tolist()
        sorted_indices=np.sort(topic_dist)[::-1].tolist()
        probabilities=[(words[x] ,sorted_indices[x]) for x in range(n_top_words)]
        topics.append((i, probabilities))
    #print(topics)

    fig, axes = plt.subplots(2, 3, figsize=(20,10), sharex=True, sharey=True)
   
    graph_titles=['Literature and Writing','Contemporary Studies','STEM','Social Sciences','Art']

    for i, ax in enumerate(axes.flatten()):
        if (i==5):
            continue
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=200)
        plt.gca().imshow(cloud)
        if labeled==False:
            plt.gca().set_title('Topic: ' + str(i), fontdict=dict(size=16))
        else:
            plt.gca().set_title('Topic: ' + graph_titles[i], fontdict=dict(size=16))
        plt.gca().axis('off')
    
    axes[-1,-1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    if labeled==False:
        plt.savefig('word_clouds_'+str(n_top_words)+'.png', bbox_inches='tight')
    else:
        plt.savefig('word_clouds_'+str(n_top_words)+'_labeled.png', bbox_inches='tight')
    plt.show()

def word_cloud_all(model, vocab, stop_words, n_top_words, mask_image):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    
    mask = np.array(Image.open(mask_image))
    
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=6000,
                      height=5000,
                      max_words=n_top_words*len(model.doc_topic_),
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0,
                      mask=mask)
    topics=[]
    temp=[]
    topic_word=model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        words=np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1].tolist()
        temp.append(words)
        sorted_indices=np.sort(topic_dist)[::-1].tolist()
        topics.extend([(words[x] ,sorted_indices[x]) for x in range(n_top_words)])
    #print(topics)
    colors_dict={}
    for i, row in enumerate(temp):
        colors_dict[cols[i]]=row
    #print(colors_dict)
    default_color='white'
    grouped_color_func = SimpleGroupedColorFunc(colors_dict, default_color)
    cloud.generate_from_frequencies(dict(topics), max_font_size=300)
    cloud.recolor(color_func=grouped_color_func)

    plt.figure(figsize=(20,10))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('word_clouds_all_'+str(n_top_words)+'.png', bbox_inches='tight')
    plt.show()

def word_cloud_courses(model, vocab, stop_words, n_top_words, labeled):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=n_top_words,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    
    with open('clustered_classes.csv') as csv_file:
        reader = csv.reader(csv_file)
        old_dict = dict(reader)
    #print(old_dict)
    data, classes=read_file()
    data=model.doc_topic_.tolist()
    max_doc_topic=[max(x) for x in data]
    topics=[]

    temp=list(old_dict.values())[:-1]
    temp=[ast.literal_eval(x) for x in temp]
    for i, row in enumerate(temp):
        probabilities=[]
        for j in row:
            probabilities.append((j, max_doc_topic[classes.index(j)]))
        topics.append((i, probabilities))
    #print(topics)
    graph_titles=['Literature and Writing','Contemporary Studies','STEM','Social Sciences','Art'] 

    fig, axes = plt.subplots(2, 3, figsize=(20,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        if (i==5):
            continue
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=200)
        plt.gca().imshow(cloud)
        if labeled==False:
            plt.gca().set_title('Topic: ' + str(i), fontdict=dict(size=16))
        else:
            plt.gca().set_title('Topic: ' + graph_titles[i], fontdict=dict(size=16))
        plt.gca().axis('off')

    axes[-1,-1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    if labeled==False:
        plt.savefig('class_word_clouds_'+str(n_top_words)+'.png', bbox_inches='tight')
    else:
        plt.savefig('class_word_clouds_'+str(n_top_words)+'_labeled.png', bbox_inches='tight')
    plt.show()


def main():
    with open('guidedlda_model.pickle', 'rb') as file_handle:
        model = pickle.load(file_handle)
    
    vocab, stopwords=print_topics(model, printing=False)
    #word_cloud_courses(model, vocab, stopwords, 10, labeled=False)
    #word_clouds(model, vocab, stopwords, 100, labeled=True)
    #word_cloud_all(model, vocab, stopwords, 100, mask_image="simons_rock_alpaca.jpeg")
    #guided_topic_graph(model.doc_topic_.tolist(), model)

if __name__ == "__main__":
    main()



