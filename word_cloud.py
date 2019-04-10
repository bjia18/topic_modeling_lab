from __future__ import print_function
warned_of_error = False
import csv
from nltk.corpus import stopwords
import pygame
import simplejson
from pytagcloud import create_tag_image, make_tags

def create_cloud (oname, words,maxsize=120, fontname='Lobster'):
    '''Creates a word cloud (when pytagcloud is installed)
    Parameters
    ----------
    oname : output filename
    words : list of (value,str)
    maxsize : int, optional
        Size of maximum word. The best setting for this parameter will often
        require some manual tuning for each input.
    fontname : str, optional
        Font to use.
    '''

    # gensim returns a weight between 0 and 1 for each word, while pytagcloud
    # expects an integer word count. So, we multiply by a large number and
    # round. For a visualization this is an adequate approximation.


    #words = [(w,int(v*10000)) for w,v in words]
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, oname, size=(1800, 1200), fontname=fontname)


#example of using word cloud for first 10 rows of courses
def main():
    word_counts = {}
    stopword_nltk = stopwords.words('english')
    with open('course-descriptions2.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        count = 0
        for row in csvreader:
            count += 1
            if count > 10:
                break
            text = row[2]
            clean_chars = [c.lower() if c.isalpha() else ' ' for c in text]
            text = "".join(clean_chars)
            words = text.split()
            for w in words:
                if w not in stopword_nltk:
                    prev_count = word_counts.get(w,0)
                    word_counts[w] = prev_count+1

    word_counts = [(w,count/10) for w,count in word_counts.items()]
    create_cloud('cloud_courses.png', word_counts)


if __name__ == "__main__":
    main()

