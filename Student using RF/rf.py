import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, model_selection
import pandas as pd

df = pd.read_csv('G:\py\Student using RF\ISE14_2 - Copy.csv')
df.replace('?', -99999, inplace = True)
df.drop(['Name', 'Result', 'USN'], 1, inplace=True)
X = np.array(df.drop(['FLorNo'],1))
y = np.array(df['FLorNo'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)
clf = RandomForestRegressor()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print (accuracy)

prediction_table = pd.read_csv('G:\py\Student using RF\ISE14_3 - Copy.csv')
prediction_table.replace('NaN', -99999, inplace=True)
prediction_table.drop(['Name','USN'], 1, inplace=True)
example = np.array(prediction_table)
prediction = clf.predict(example)
print(prediction)

#count_fail=0
#count_pass=0
#for i in prediction:
#    if i in prediction != 0:
#        count_fail+=count_fail
#    else:
#        count_pass+=count_pass
#print (count_fail)#, (count_pass), (count_pass/i)

#0.544380952381
#[ 0.1  0.   0.2  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#  0.1  0.   0.   1.   0.   0.   0.   0.   0.   0.   0.   0.2  0.   0.   0.
#  0.   0.   0.   0.   0.2  0.   0.   0.   0.2  0.   0.   0.3  0.   0.   0.
#  0.   0.   0.   0.2  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.4  0.
#  0.   0.   0.   0. ]
#>>> 54/64
#0.84375
