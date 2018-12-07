# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm
#
# xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
# #linspace 表示线性地生成ndarray，前两个参数表示起始区间，第三个表示生成的元素个数
# #类似的是arange，不过arange的第三个参数表示步长
# #meshgrid,用于生成坐标，不过例子中一般是用来画图的所以我们先只考虑二维的情况
# #meshgrid(a,b)返回的xx,yy，xx的每一行都是向量a，重复len(b)次，yy的每一列都是向量b，重复len(a)次
#
# np.random.seed(0)
# X = np.random.randn(300, 2)#randn用于生成标准正态分布的数据，里面的两个数表示生成矩阵的大小
# Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)#求异或
#
# # fit the model
# clf = svm.NuSVC()#生成一个NuSVC的estimator不过还没有进行训练
# clf.fit(X, Y)#使用数据进行训练，大多数estimator都有个fit函数
#
# # plot the decision function for each datapoint on the grid
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# #ravel()的作用是把多维数组拉伸成一维数组，c_的作用是将两个ndarray的相同位置处的
# #元素合在一起，对于这里的情况就是合并出一个坐标来，例子：
# #>>np.c_[np.array([1,2,3]), np.array([4,5,6])]
# #>>array([[1, 4],
# #       [2, 5],
#  #      [3, 6]])
#  #注意c_后面直接跟的是[]
#  #decision_function返回的是点到超平面的有向距离（带符号的）
#
# #最后reshape(（tuple）)的作用是再把一维数组变成括号里面的元组的大小
#
# plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',origin='lower', cmap=plt.cm.PuOr_r)
# #画图，cmap参数决定是用什么样的颜色风格
# contours = plt.contour(xx, yy, Z, levels=[0])
# # from matplotlib.colors import ListedColormap
# # custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
# # contours = plt.contourf(xx, yy, Z, cmap=custom_cmap)
# #contour画的是“云图”的线，我理解的就是等高线，其中levels参数，A list of floating point numbers indicating the level curves to draw, in increasing order; e.g., to draw just the zero contour pass levels=[0]。
# #这个数值我理解的就是在SVM确定后wx+b=0这个等式右边的值，0就代表超平面，+1和-1代表经过支持向量的平面，这个经过试验我发现应该是没错的。
# plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors='k')
# #画散点图，对于其中的参数c，我没看懂API的解释（c can be a single color format string, or a sequence of color specifications of length N, or a sequence of N numbers to be mapped to colors using the cmap and norm specified via kwargs (see below). ）但是经过试验发现不能把不同类别的点标记成不同颜色了。
# plt.xticks(())
# plt.yticks(())
# plt.axis([-3, 3, -3, 3])
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X=np.c_[(.4,-.7),(-1.5,-1),(-1.4,-.9),(-1.3,-1.2),(-1.1,-.2),(-1.2,-.4),(-.5,1.2),(-1.5,2.1),(1,1),
         (1.3,.8),(1.2,.5),(.2,-2),(.5,-2.4),(.2,-2.3),(0,-2.7),(1.3,2.1)].T
#这里的c_把各个元组变成了[]，整个X变成了矩阵
Y = [0]*8+[1]*8
#Y是一个list，8个0，8个1
print(Y)
fignum = 1
plt.figure(1, figsize=(12,5))
for kernel in ('linear','poly','rbf'):
#这是对三种核分别计算并画图
    clf = svm.SVC(kernel =kernel, gamma=2)
    # SVC的介绍见下方
    clf.fit(X, Y)
    # 画图区域
    plt.subplot(1,3,fignum)
    # 绘制所有点
    plt.scatter(X[:,0], X[:,1], c = Y, zorder=10, cmap=plt.cm.Paired, edgecolors = 'k')
    # 画出支持向量
    plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], marker='s', s = 100, facecolors ='none', zorder = 10, edgecolors = 'k')
    plt.title(kernel)
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    #同样也是用来生成网格的与meshgrid类似，x_min:x_max:200j用于生成array，好像只能和mgrid连用,这点需要注意
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    # plt.pcolormesh(XX, YY, Z > 0, cmap = plt.cm.Paired)
    #pcolormesh:Plot a quadrilateral mesh.参数C may be a masked array
    plt.contour(XX, YY, Z, levels = [-1, -0.5, 0, 0.5, 1])
    #这里画出了三条线，分别是wx+b等于-0.5，0，0.5三种
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    fignum = fignum+1
plt.show()

