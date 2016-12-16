import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv('G:\py\Student using KNN\ISE14_2 - Copy.csv')
#df_test = pd.read_csv('G:\py\Student using KNN\ISE14_3 - Copy.csv')
df_train.replace('?', -99999, inplace=True)
df_train.drop(['Name','Result','USN'], 1, inplace=True)
#df_test.drop(['Name'], 1, inplace=True)
pass_fail_original = np.array(df_train['FLorNo'])
#row_count = sum(1 for row in pass_fail_original)  # fileObject is your csv.reader
#print(row_count)
#pass_row_count=0
#for i in pass_fail_original:
#    if 'FLorNo' == '0':
#        pass_row_count+=pass_row_count
#print(pass_row_count)
    
X = np.array(df_train.drop(['FLorNo'],1))
#Xtest = np.array(df_test.drop(['FLorNo'],1))
y = np.array(df_train['FLorNo'])
#ytest = np.array(df_train['FLorNo'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

#PREDICTION TIME!

#USING TABLE
prediction_table = pd.read_csv('G:\py\Student using KNN\ISE14_3 - Copy.csv')
prediction_table.replace('NaN', -99999, inplace=True)
prediction_table.drop(['Name','USN'], 1, inplace=True)
example = np.array(prediction_table)
prediction = clf.predict(example)
print(prediction)

print(df_train.describe())
df_train.hist()
plt.show()
scatter_matrix(df_train.drop(['USN', 'Name'], 1))
plt.show()
data.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
plt.show()


#Printing to another file.

#import csv
#with open('test.csv','w',newline='') as fp:
##    a = csv.writer(fp,delimiter=',')
#    data=[['stock','Sales'],
#          ['100','24']]
#   a.writerows(data)
#USING EXAMPLES
#example_pass = np.array([18,59,78,17,43,63,20,66,66,21,63,84,18,67,75,23,37,70,23,45,68,20,63,73,594])
#example_fail = np.array([1,5,7,7,4,6,2,6,6,2,6,8,1,6,7,2,3,7,2,4,6,2,6,7,59])
#prediction = clf.predict(example_fail)
#print(prediction)

#now we need the requirement. i.e. pass percentage of the predicted data.
#import csv

#file1 = csv.reader(open("G:\py\Student using KNN\ISE14_3 - Copy.csv"), delimiter)
#header1 = file1.next()
#for '1e', '2e', '3e', '4e', '5e', '6e', '7e', '8e' in prediction_table:
#    if ('1e' > '34'):
#        table_content = 0
#    else:
#        table_content = 1
        

