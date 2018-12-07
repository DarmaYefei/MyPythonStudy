# import numpy as np
# import random
# import matplotlib.pyplot as plt
# def gradientDescent(x, y, alpha, numIterations):
#     xTrans = x.transpose()
#     m, n = np.shape(x)
#     theta = np.ones(n)
#     for i in range(0, numIterations):
#         hwx = np.dot(x, theta)
#         loss = hwx - y
#         cost = np.sum(loss ** 2) / (2 * m)
#         # print("Iteration %d | Cost: %f " % (i, cost))
#         gradient = np.dot(xTrans, loss) / m
#         theta = theta - alpha * gradient
#     return theta
#
# def genData(numPoints, bias, variance):
#     x = np.zeros(shape=(numPoints, 2))
#     y = np.zeros(shape=numPoints)
#     for i in range(0, numPoints):
#         x[i][0] = 1
#         x[i][1] = i
#         y[i] = (i + bias) + random.uniform(0, 1) * variance
#     return x, y
#
# def plotData(x,y,theta):
#     plt.scatter(x[...,1],y)
#     plt.plot(x[...,1],[theta[0] + theta[1]*xi for xi in x[...,1]])
#     plt.show()
#
# x, y = genData(20, 25, 10)
# print(x)
# print(y)
# iterations= 10000
# alpha = 0.001
# theta=gradientDescent(x,y,alpha,iterations)
# plotData(x,y,theta)

# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# # from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# X, y = datasets.make_classification(n_samples=100,n_classes=3,n_features=5, n_informative=3, n_redundant=0,random_state=42)
# # Binarize the output
# y = label_binarize(y, classes=[0, 1, 2])
# n_classes = y.shape[1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True, ))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, label='ROC AUC %0.2f' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="best")
# plt.show()

import numpy as np

x = np.random.normal(0, 1, size=(4, 2))
y = np.array([0, 1, 0, 1])

print(x[y==1], x[y==1, 0], x[y==1, 1], sep='\n')