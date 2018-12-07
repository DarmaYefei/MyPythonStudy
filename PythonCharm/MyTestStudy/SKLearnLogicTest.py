import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)

DotNum = 200
# 生成正态分布范围内的DotNum个点(a,b)
# 参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
# 参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
# 参数size(int 或者整数元组)：输出的值赋在shape里，默认为None
scale = 1
X = np.random.normal(0, scale, size=(DotNum, 2))
# a^2+b<1的点设置为1类，其它为0类
y = np.array((X[:,0]**2 + X[:,1]) < scale, dtype='int')
# 随机抽取 10% 个样本，让其分类为 1，相当于认为更改数据，添加噪音
for _ in range(int(DotNum/10)):
    y[np.random.randint(DotNum)] = 1
plt.figure(figsize=(15, 6))
plt.subplot(1, 5, 1)
# 绘制分类，y==0为一个布尔数组，取出其对应序列的X第一列代表a，第二列代表b，绘制散点图
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])

from sklearn.model_selection import train_test_split
# 将数据进行拆分测试库和训练库
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.linear_model import LogisticRegression
# 逻辑回归， 默认参数设置C=1.0，正则化penalty='L2'
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)

def plot_decision_boundary(model, axis):
    # meshgrid(m, n) 扩展成为m * n的矩阵，假设m为长度m的一位数组，假设n为长度n的一位数组
    # linspace均分指令，即将[a, b]区间划分为N个数字
    # reshape(-1, 1)即不知道行数，变成一列，行数视数据个数而定
    x0, x1 = np.meshgrid(np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    # 颜色分别为第一类，分界线，第二类
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    # contourf绘制轮廓并填充
    plt.contourf(x0, x1, zz, cmap=custom_cmap)


plt.subplot(1, 5, 2)
plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def polynomialLogisticRegression(degree):
    # from sklearn.preprocessing import PolynomialFeatures来进行特征的构造。
    # 它是使用多项式的方法来进行的，如果有a，b两个特征，那么它的2次多项式为（1, a, b, a ^ 2, ab, b ^ 2）。
    # PolynomialFeatures有三个参数
    # degree：控制多项式的度
    # interaction_only： 默认为False，如果指定为True，那么就不会有特征自己和自己结合的项，上面的二次项中没有a ^ 2和b ^ 2。
    # include_bias：默认为True。如果为True的话，那么就会有上面的1
    # from sklearn.preprocessing import StandardScaler
    # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。 StandardScaler对每列分别标准化，因为shape of data: [n_samples, n_features]
    # sklean提供的pipeline来将多个学习器组成流水线，通常流水线的形式为：
    # 将数据标准化的学习器---特征提取的学习器---执行预测的学习器
    # 除了最后一个学习器之外，前面的所有学习器必须提供transform方法，该方法用于数据转化（例如：归一化，正则化，以及特征提取
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


# 使用管道时，先生成实例的管道对象，再进行fit；
poly_log_reg = polynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)

plt.subplot(1, 5, 3)
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])


def polynomialLogisticRegression(degree, C, penalty='l2'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C, penalty=penalty))
    ])


poly_log_reg2 = polynomialLogisticRegression(degree=2, C=1)
poly_log_reg2.fit(X_train, y_train)
plt.subplot(1, 5, 4)
plot_decision_boundary(poly_log_reg2, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])

plt.subplot(1, 5, 5)
poly_log_reg4 = polynomialLogisticRegression(degree=2, C=1, penalty='l1')
poly_log_reg4.fit(X_train, y_train)

plot_decision_boundary(poly_log_reg4, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()