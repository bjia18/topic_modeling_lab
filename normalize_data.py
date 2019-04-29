from preprocess_file import *
from print_model import *
import decimal

def normalize_values(topics): 
    min_num=None
    max_num=None
    
    #normalize values to 0~1
    for i in topics:
        temp=min(i)
        temp2=max(i)
        if min_num==None or min_num>temp:
            min_num=temp
        if max_num==None or max_num<temp2:
            max_num=temp2
    print("min: "+str('{:.20f}'.format(min_num))+" max: "+str(max_num))
    
    #scale to 0~10^7 (round to integers)
    for i in range(len(topics)):
        for j in range(len(topics[i])):
            topics[i][j]=(topics[i][j]-min_num)/(max_num-min_num)
            topics[i][j]*=10**7
            topics[i][j]=round(topics[i][j])
            #print(topics[i][j])
    min_num=0
    max_num=None
    
    #check for decimals
    for i in topics:
        temp=min(i)
        temp2=max(i)
        if temp<1 or temp<min_num:
            min_num=temp
        if max_num==None or max_num<temp2:
            max_num=temp2
    print("min: "+str(min_num)+" max: "+str(max_num))
   
    with open('doc_topic_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in topics:
            writer.writerow(i)

def main():
    with open('guidedlda_model.pickle', 'rb') as file_handle:
        model = pickle.load(file_handle)
    #print(model.doc_topic_)
    topic_count=model.doc_topic_.tolist()
    normalize_values(topic_count)

if __name__ == "__main__":
    main()
 
