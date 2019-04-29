import clusters as clus
func=clus.pearson
best_k=5

import matplotlib.pyplot as plt
import numpy as np
from preprocess_file import *
from print_model import *
import bisect_k
import hierarchical
#import word_cloud
from sse_and_centroid import*

def read_file():
    data=[]
    classes=[]
    columns=True
    with open('course-descriptions2.csv', errors='replace') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            if columns==True:
                columns=False
                continue
            classes.append(row[0]+' '+row[1])
   
    with open('doc_topic_data.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            integers=[int(i) for i in row]
            data.append(integers)
    
    return data, classes

def elbow_method(data):
    # Data for plotting
    max_range=range(1,25)
    x_k=[]
    y_sse=[]
    for i in max_range:
        #raw_clusters = clus.kcluster(data, distance=func, k=i)
        raw_clusters=b_k_test(data, i) 
        clusters=bisect_k.eliminate(raw_clusters)
        x_k.append(i)
        y_sse.append(sse(clusters, data))

    fig, ax = plt.subplots()
    ax.plot(x_k, y_sse)

    ax.set(xlabel='k', ylabel='sse',title='bisect k-means')
    fig.savefig("elbow_chart.png")
    plt.show()

def test_metrics(data):
    sse_metrics=[]
    metrics=['manhattan','euclidean','cosine','pearson','tanimoto','bisect']
    clusters=bisect_k.eliminate(clus.kcluster(data, distance=clus.manhattan, k=best_k))
    sse_metrics.append(sse(clusters, data))
    clusters=bisect_k.eliminate(clus.kcluster(data, distance=clus.euclidean, k=best_k))
    sse_metrics.append(sse(clusters, data))
    clusters=bisect_k.eliminate(clus.kcluster(data, distance=clus.cosine, k=best_k))
    sse_metrics.append(sse(clusters, data))
    clusters=bisect_k.eliminate(clus.kcluster(data, distance=clus.pearson, k=best_k))
    sse_metrics.append(sse(clusters, data))
    clusters=bisect_k.eliminate(clus.kcluster(data, distance=clus.tanimoto, k=best_k))
    sse_metrics.append(sse(clusters, data))
    clusters=bisect_k.eliminate(b_k_test(data, best_k))
    sse_metrics.append(sse(clusters, data))
    
    fig, ax = plt.subplots()
    ax.plot(metrics, sse_metrics)

    ax.set(xlabel='metrics', ylabel='sse',title='measure distance metrics')
    fig.savefig("metrics.png")
    plt.show()

def b_k_test(data, test_k):
    x=test_k+2
    raw_clusters = bisect_k.bisect([list(range(len(data)))], data, distance=clus.pearson, k=x)
    clusters = []
    for i in range(test_k):
        if len(raw_clusters[i]) == 0:
            continue
        clusters.append(raw_clusters[i])
    return clusters

def b_k(data, countries):
    tests=[]
    for j in range(501):
        test={}
        raw_clusters = bisect_k.bisect([list(range(len(data)))], data, distance=clus.pearson, k=best_k+2)
        clusters = []
        for i in range(best_k):
            if len(raw_clusters[i]) == 0:
                continue
            clusters.append(raw_clusters[i])
            test['cluster {}:'.format(i + 1)]=[countries[r] for r in clusters[i]]
        test['sse']=sse(clusters, data)
        tests.append(test)
        print('total: '+str(j))
    best_sse=50
    pos=None
    for i in range(len(tests)):
        if tests[i]['sse']<best_sse:
            best_sse=tests[i]['sse']
            pos=i
    if pos!=None:
        mydict=tests[pos]
        with open('clustered_classes.csv') as csv_file:
            reader = csv.reader(csv_file)
            old_dict = dict(reader)
        if float(old_dict['sse'])>mydict['sse']:
            with open('clustered_classes.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for key, value in mydict.items():
                    writer.writerow([key, value])
        print('sse: '+str(mydict['sse']))

def main():
    data, classes=read_file()
    #elbow_method(data)
    #test_metrics(data)
    b_k(data, classes)
    #hierarchical.hier(data,classes)
    '''raw_clusters = clus.kcluster(data, distance=func, k=best_k)
    clusters = []
    class_clusters = []
    for i in range(best_k):
        if len(raw_clusters[i]) == 0:
            continue
        clusters.append(raw_clusters[i])
        print('cluster {}:'.format(i + 1))
        print([classes[j] for j in raw_clusters[i]])
        class_clusters.append([classes[j][1] for j in raw_clusters[i]])
    print("sse: " + str(sse(clusters, data)))'''

    #word_cloud.cloud(clusters, data)

if __name__ == "__main__":
    main()
