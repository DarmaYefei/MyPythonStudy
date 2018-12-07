from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import numpy as np
# from sklearn import cross_validation

iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
X, y = X_iris[:, :2], y_iris

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)

print(X_train.shape, X_test.shape)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    # plt.scatter(xs, ys, c=colors[i])

# plt.legend(iris.target_names)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.show()

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, tol=None)
clf.fit(X_train, y_train)

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
xs = np.arange(x_min, x_max, 0.5)
# for i in [0, 1, 2]:
#     ax1 = picture.add_subplot(1, 3, i + 1)
#     ax1.set_xlabel('Sepal length')
#     ax1.set_ylabel('Sepal width')
#     ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
#     ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
#     ax1.plot(xs, ys)

fig, axes = plt.subplots(1, 3)
for i in range(3):
    axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    axes[i].plot(xs, ys)

plt.show()
