import clusters as clus

def sse(clusters, data):
    sum = 0
    for c in clusters:
        centroid = calc_centroid(c, data)
        for country in c:
            sum+=pow(clus.pearson(data[country], centroid), 2)
    return sum

def calc_centroid(c, data):
    centroid = [0 for i in range(len(data[0]))]
    for i in c:
        for j in range(len(data[0])):
            centroid[j] += data[i][j]

    centroid = [round(centroid[i]/len(c)) for i in range(len(centroid))]
    return centroid

def eliminate(raw_clusters):
    clusters=[]
    for i in range(len(raw_clusters)):
        if len(raw_clusters[i]) != 0:
            clusters.append(raw_clusters[i])
    return clusters

