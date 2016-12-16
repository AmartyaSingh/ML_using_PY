import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

data = pd.read_csv('tweets_data.data.txt')
data.replace('?', -99999, inplace = True)
data.drop(['USER_ID'], 1, inplace = True)
X = np.array(data.drop(['market_pred_by_user'], 1))
y = np.array(data['market_pred_by_user'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.9)

#RANDOM_FOREST_CLASSIFIER
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
accuracy_rf = clf.score(X_test, y_test)
abs_val = (accuracy_rf)
print ("RF:", abs_val)
#print(data.describe())
#data.hist()
#plt.show()
#scatter_matrix(data)
#plt.show()
#data.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
#plt.show()
prediction_table = pd.read_csv('tweets_data_pred.data.txt')
prediction_table.replace('NaN', -99999, inplace=True)
prediction_table.drop(['USER_ID', 'actual_market_behavior'], 1, inplace=True)
example_RF = np.array(prediction_table)
prediction_RF = clf.predict(example_RF)
print(prediction_RF)

#K_NEAREST_NEIGHBORS_CLASSIFIER
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy_knn = clf.score(X_test, y_test)
print ("KNN:", accuracy_knn)
example_KNN = np.array(prediction_table)
prediction_KNN = clf.predict(example_KNN)
print(prediction_KNN)

#SUPPORT_VECTOR_MACHINE_CLASSIFIER
clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy_svm = clf.score(X_test, y_test)
print ("SVM", accuracy_svm)
example_SVM = np.array(prediction_table)
prediction_SVM = clf.predict(example_SVM)
print(prediction_SVM)

#count_correct_prediction=0
#count_incorrect_prediction=0
#data = pd.DataFrame()
#data_1 = np.where(data['market_pred_by_user'] == data['actual_market_behavior'], count_correct_prediction = 1, count_incorrect_prediction = 1)
#print ("count_correct_prediction:", count_correct_prediction)
#print ("count_incorrect_prediction:", count_incorrect_prediction)


