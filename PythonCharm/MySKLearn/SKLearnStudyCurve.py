import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

n_dots = 200

X = np.linspace(0, 1, n_dots)
y = np.sqrt(X) + 0.2 * np.random.rand(n_dots) - 0.1

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


def polynomial_model(degree = 1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("poly", polynomial_features),
                         ("linear", linear_regression)])
    return pipeline


def plot_learning_curve(plt, estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    # plt.fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, hold=None, data=None, **kwargs)
    # x - array(length N) 定义曲线的x坐标
    # y1 - array(length N) or scalar定义第一条曲线的y坐标
    # y2 - array(length N) or scalar定义第二条曲线的y坐标
    # where - array of bool(length N), optional, default: None
    # 排除一些（垂直）区域被填充。
    # 注：我理解的垂直区域，但帮助文档上写的是horizontal regions
    # 也可简单地描述为plt.fill_between(x，y1，y2，where = 条件表达式, color = 颜色，alpha = 透明度)
    # " where = "可以省略，直接写条件表达式
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# 为了让学习曲线更平滑，交叉验证数据集的得分计算 10 次，每次都重新选中 20% 的数据计算一遍
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# sklearn.model_selection.ShuffleSplit类用于将样本集合随机“打散”后划分为训练集、测试集(可理解为验证集，下同)，类申明如下
# class sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=’default’, train_size=None, random_state=None)
# 参数：
# n_splits: int, 划分训练集、测试集的次数，默认为10
# test_size: float, int, None, default = 0.1； 测试集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示测试集占总样本的比例；
# 该值为整型值时，表示具体的测试集样本数量；train_size不设定具体数值时，该值取默认值0.1，train_size设定具体数值时，test_size取剩余部分
# train_size: float, int, None； 训练集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示训练集占总样本的比例；
# 该值为整型值时，表示具体的训练集样本数量；该值为None(默认值)时，训练集取总体样本除去测试集的部分
# random_state: int, RandomState
# instance or None；随机种子值，默认为None

titles = ['Learning Curves (Under Fitting)',
          'Learning Curves',
          'Learning Curves (Over Fitting)']
degrees = [1, 3, 10]

plt.figure(figsize=(15, 5))
for i in range(len(degrees)):
    plt.subplot(1, 3, i + 1)
    plot_learning_curve(plt, polynomial_model(degrees[i]), titles[i], X, y, ylim=(0.75, 1.01), cv=cv)

plt.show()
