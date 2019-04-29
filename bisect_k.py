import clusters as clus
from sse_and_centroid import *

def eliminate(raw_clusters):
    clusters=[]
    for i in range(len(raw_clusters)):
        if len(raw_clusters[i]) != 0:
            clusters.append(raw_clusters[i])
    return clusters


def bisect (clusters, data, distance=clus.euclidean, k=4):
    if len(clusters) == k:
        return clusters
    max_sse = 0
    c_index = 0

    clusters=eliminate(clusters)
    for i in range(len(clusters)):
        c= clusters[i]
        score = 0
        
        centroid = calc_centroid(c, data)

        for country in c:
            score += pow(distance(data[country], centroid), 2)

        if score > max_sse:
            max_sse = score
            c_index = i

    indexes = []
    for i in clusters[c_index]:
        indexes.append(i)

    new_clusters = clus.kcluster([data[i] for i in clusters.pop(c_index)], distance=distance, k=2)
    for c in new_clusters:
        for i in range(len(c)):
            c[i] = indexes[c[i]]

    return bisect(clusters + new_clusters, data, distance=distance, k=k)

