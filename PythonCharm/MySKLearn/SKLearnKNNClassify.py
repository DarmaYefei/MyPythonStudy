import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# 生成数据
centers = [[-2, 2], [2, 2], [0, 4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.6)

# sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0),
# shuffle=True, random_state=None)
# n_samples: int, optional (default=100)
# The total number of points equally divided among clusters.
# 待生成的样本的总数。
# n_features: int, optional (default=2)
# The number of features for each sample.
# 每个样本的特征数。
# centers: int or array of shape [n_centers, n_features], optional (default=3)
# The number of centers to generate, or the fixed center locations.
# 要生成的样本中心（类别）数，或者是确定的中心点。
# cluster_std: float or sequence of floats, optional (default=1.0)
# The standard deviation of the clusters.
# 每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]。
# center_box: pair of floats (min, max), optional (default=(-10.0, 10.0))
# The bounding box for each cluster center when centers are generated at random.
# shuffle: boolean, optional (default=True)
# Shuffle the samples.
# random_state: int, RandomState instance or None, optional (default=None)
# If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the
# random number generator; If None, the random number generator is the RandomState instance used by np.random.

# 画出数据
plt.figure(figsize=(8, 6))
c = np.array(centers)
# 画出样本
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
# 画出中心点
plt.scatter(c[:, 0], c[:, 1], s=100, marker='*', c='orange')

# 模型训练
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)
# 进行预测
X_sample = [-1, 2]
X_sample = np.array(X_sample).reshape(1, -1)
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample, return_distance=False)
print(neighbors)
plt.scatter(X_sample[0][0], X_sample[0][1], marker="x", s=100, cmap='cool')    # 待预测的点
for i in neighbors[0]:
    # 预测点与距离最近的 5 个样本的连线
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 'k--', linewidth=0.6)

plt.show()
