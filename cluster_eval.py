from preprocess_file import *
from print_model import *
import clusters as clus
import class_clustering as cc
import seaborn as sn
import ast
import math
import sys
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn import datasets

np.set_printoptions(threshold=sys.maxsize)
sn.set(rc={'figure.figsize':(15,10)})

def confusion_matrix(temp):
    labels=[]
    n=0
    for i in temp:
        for j in i:
            labels.append(j.split(' ')[0])
            n+=1
    
    labels = list(dict.fromkeys(labels))
    matrix = [[0 for i in range(len(labels))] for j in range(len(temp))]
    classes_counts=[0 for i in range(len(labels))]
    clusters_counts=[0 for i in range(len(temp))]
    
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            matrix[i][labels.index(temp[i][j].split(' ')[0])]+=1
            classes_counts[labels.index(temp[i][j].split(' ')[0])]+=1
        clusters_counts[i]+=len(temp[i])
    
    probabilities=[]

    for i in range(len(temp)):
        row=[]
        for j in range(len(labels)):
            row.append(matrix[i][j]/clusters_counts[i])
        probabilities.append(row)
    #print(probabilities)
    cluster_entropy=[0 for i in range(len(temp))]
    cluster_purity=[0 for i in range(len(temp))]

    for i in range(len(probabilities)):
        cluster_purity[i]=max(probabilities[i])
        for j in range(len(probabilities[i])):
            if probabilities[i][j]!=0:
                cluster_entropy[i]+=-(probabilities[i][j]*math.log2(probabilities[i][j]))

    entropy=0
    purity=0
    for i in range(len(cluster_entropy)):
        entropy+=(clusters_counts[i]/n)*cluster_entropy[i]
        purity+=(clusters_counts[i]/n)*cluster_purity[i]
    
    worst_entropy=math.log2(len(labels))
    #print(cluster_purity)
    #print('purity: '+str(purity))
    #print('entropy: '+str(entropy)+' out of '+str(worst_entropy)+'\npercentage: '+str(entropy/worst_entropy*100)+'%')

    #print(probabilities)
    percentages=[[0 for i in range(len(temp))] for j in range(len(labels))]
    for i in range(len(percentages)):
        for j in range(len(percentages[i])):
            percentages[i][j]=round((matrix[j][i]/classes_counts[i])*100, 2)

    print(percentages)
    
    cluster_labels=['Literature and Writing','Contemporary Studies','STEM','Social Sciences','Art and Languages']

    df_cm = pd.DataFrame(percentages, index = [i for i in labels],
                  columns = [i for i in cluster_labels])
    yticks = df_cm.index
    xticks = df_cm.columns

    heat=sn.heatmap(df_cm,linewidth=0,yticklabels=yticks,xticklabels=xticks, annot=True, cmap='Blues', fmt='g')

    # This sets the yticks "upright" with 0, as opposed to sideways with 90.
    plt.xticks(rotation=0)
    plt.title('Percentages of Classes in Clusters')
    fig=heat.get_figure()
    fig.savefig('percentage_matrix.png', bbox_inches='tight')

    plt.show()
    #print(cluster_entropy)
    #print(matrix)
    #print(classes_counts)
    #print(clusters_counts)

def distance_matrix(temp, data, classes):

    target=[]
    for i in classes:
        for sub_list in temp:
            if i in sub_list:
                target.append(temp.index(sub_list))
    data=[x for _,x in sorted(zip(target,data))]
    target.sort()
     
    np_data=np.array([np.array(x) for x in data])
    dist_mat = squareform(pdist(np_data, 'cosine'))
    #print(len(dist_mat)) 
    inc_mat=[]
    for i in range(len(data)):
        temp2=[]
        for j in range(len(data)):
            if target[i]==target[j]:
                temp2.append(1)
            else:
                temp2.append(0)
        inc_mat.append(temp2)
    inc_mat=np.array([np.array(x) for x in inc_mat])
    corr=np.corrcoef(dist_mat,inc_mat)#[1, 0]
    #coef=1 - cdist(dist_mat, inc_mat, metric='correlation')#[1,0]
    
    #print(corr.shape)
    with sn.axes_style("white"):
        ax = sn.heatmap(dist_mat, vmax=1, square=True,  cmap="YlGnBu")
        plt.savefig('heatmap.png', bbox_inches='tight')
        plt.show()

def main():
    data, classes=cc.read_file()
    with open('clustered_classes.csv') as csv_file:
        reader = csv.reader(csv_file)
        old_dict = dict(reader)
    temp=list(old_dict.values())[:-1]
    temp=[ast.literal_eval(x) for x in temp]
    
    confusion_matrix(temp)
    #distance_matrix(temp, data, classes)

if __name__ == "__main__":
    main()



