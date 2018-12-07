import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()

start = time.clock()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.clock()-start, train_score, cv_score))


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_feature = LinearRegression()
    pipeline = Pipeline([('polynomial', polynomial_features),('linear', linear_feature)])
    return pipeline

model = polynomial_model(2)
start = time.clock()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.clock()-start, train_score, cv_score))

from sklearn.model_selection import ShuffleSplit
from CommomUtils import plot_learning_curve

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(10, 8))
title = 'Learning Curve (degree = {0})'
degree = [1, 2, 3]

start = time.clock()
for i in degree:
    plt.subplot(1, 3, i)
    plot_learning_curve(plt, polynomial_model(i), title.format(i), X, y, ylim=(-0.1,1.1), cv=cv)

plt.show()
print('elaspe:{0:.6f}'.format(time.clock()-start))

