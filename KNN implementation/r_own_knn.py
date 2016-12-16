import numpy as np
from math import sqrt
#import matplotlib.pyplot as plt
import warnings
#from matplotlib import style
from collections import Counter
import pandas as pd
import random

#style.use('fivethirtyeight')

#dataset = {'k':[[1,2],[2,3],[3,4]], 'r':[[7,8],[8,9],[9,10]]}
#new_feature = [5,7]

#for i in dataset:
    #for ii in dataset[i]:
        #plt.scatter(ii[0], ii[1], s=100, color=i)

#plt.scatter(new_feature[0], new_feature[1])
#plt.show()

def knn(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K should be greater than len of data!')
    distances = []
    for group in data:  #data set similar to dataset with groups being 'k' and 'r'
        for features in data[group]: #the features of 'k' and 'r'
            euclid_dist = np.linalg.norm(np.array(features)-np.array(predict))  #to calc euclidian distances
            distances.append([euclid_dist, group])
            # ^ list of lists, 1st item is distances, 2nd is group then sort the list
            #taking the 3 things in list and taking the 1st in every group.

    votes = [i[1] for i in sorted(distances)[:k]]
    # ^       ^group         ^sort distances in asc order then we care only for those to k.
    print(Counter(votes).most_common(1))
    # ^ will display the group/class that is the most common, along with the majority value.
    vote_result = Counter(votes).most_common(1)[0][0]
    # ^                                      ^1st most common vote.
    return vote_result

#result = knn(dataset, new_feature, k=3)
#print(result)

df = pd.read_csv('G:\py\KNN implementation\breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.dfop(['id', 1, inplace=True])
full_data = df.astype(float)


