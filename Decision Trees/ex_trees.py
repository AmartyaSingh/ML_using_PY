from sklearn import tree

X = [[160, 15, 20], [170, 20, 25], [180, 25, 30], [190, 30, 35]]
Y = ['Female','Female','Male','Male']

#Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
predict = clf.predict([[176, 23, 28]])
print (predict)
