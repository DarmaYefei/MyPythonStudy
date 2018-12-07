import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

n_dots = 200
X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_feature = LinearRegression()
    pipeline = Pipeline([('polynomial', polynomial_features),('linear', linear_feature)])
    return pipeline


degree = [2, 5, 10, 15]
results = []
for i in range(len(degree)):
    model = polynomial_model(degree[i])
    model.fit(X, y)
    train_score = model.score(X, y)
    mse = mean_squared_error(y, model.predict(X))
    results.append({'model':model, 'degree':degree[i], 'score':train_score, 'mse':mse})

for r in results:
    print("degree: {}; train score: {}; mean squared error: {}".format(r["degree"], r["score"], r["mse"]))

from matplotlib.figure import SubplotParams

plt.figure(figsize=(16, 8), dpi=100, subplotpars=SubplotParams(hspace=0.5, wspace=0.5))
for i, r in enumerate(results):
    # i 表示索引第几个，r是内容
    fig = plt.subplot(2, 2, i+1)
    plt.xlim(-10, 10)
    plt.title("LinearRegression degree={}".format(r["degree"]), fontsize=10)
    plt.scatter(X, y, s=5, c='b', alpha=0.5)
    plt.plot(X, r["model"].predict(X), 'r-')
    x_new = np.linspace(-10, 10, 1000)
    x_new = x_new.reshape(-1, 1)
    y_new = r["model"].predict(x_new)
    # plt.plot(x_new, y_new, 'g:')

plt.show()