import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
n_dots = 80
X = 5 * np.random.rand(n_dots, 1)
y = np.cos(X).ravel()

y += 0.2 * np.random.rand(n_dots) - 0.1

k = 5
knn = KNeighborsRegressor()
knn.fit(X, y)

# 生成足够密集的点并进行预测
T = np.linspace(0, 5, 500)[:, np.newaxis]
# np.newaxis 为 numpy.ndarray（多维数组）增加一个轴
y_pred = knn.predict(T)
knn.score(X, y)

# 画出拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(X, y, c='g', label='data', s=100)         # 画出训练样本
plt.plot(T, y_pred, c='k', label='prediction', lw=4)  # 画出拟合曲线
plt.axis('tight')
plt.title("KNeighborsRegressor (k = %i)" % k)
plt.show()