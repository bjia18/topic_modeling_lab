from math import *
from sse_and_centroid import *

import clusters as clus

def all(cluster, ids):
    if cluster.id < 0:
        if cluster.left is not None:
            all(cluster.left, ids)
        if cluster.right is not None:
            all(cluster.right, ids)
    else:
        ids.append(cluster.id)


def hier(data, countries):
    clusters = clus.hcluster(data, distance=clus.pearson)
    clus.drawdendrogram(clusters, list(map(lambda x: x[1], countries)), jpeg='hierarch.jpg')
    clusters = [clusters.left.left.left, clusters.left.left.right, clusters.left.right,
                     clusters.right.left.left, clusters.right.left.right.left,
                     clusters.right.left.right.right, clusters.right.right]

    cluster_l = []

    for c in clusters:
        country_ids = []
        all(c, country_ids)
        cluster_l.append(country_ids)

    for i in range(7):
        print('cluster {}:'.format(i + 1))
        print([countries[r] for r in cluster_l[i]])

    print("SSE: " + str(sse(cluster_l, data)))
