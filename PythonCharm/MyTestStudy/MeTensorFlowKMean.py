# import tensorflow as tf
# import numpy as np
# import time
# import matplotlib
# import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
# #样本数
# N=8
# #族数
# K=3
# #最大迭代数
# MAX_ITERS = 20
# changed = True
# iters = 0
#
# centers = [(2, 10.0), (5, 8), (1, 2)]
# data=np.array([[2,10],[2,5],[8,4],[5.0,8],[7,5],[6,4.0],[1,2],[4,9]])
# features=np.array([0,1,1,1,1,2,2,2])
# fig, ax = plt.subplots()
# #s-表示标记的大小
# ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker = 'o', s = 250)
# plt.show()
#
# fig, ax = plt.subplots()
# ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker = 'o', s = 250)
# ax.scatter(data.transpose()[0], data.transpose()[1], marker = 'o', s = 100, c = features, cmap=plt.cm.coolwarm )
# plt.show()
#
# points = tf.Variable(data)
# cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))
# centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(centroids)
#
# # 为计算每点对族中心的距离，使rep_centroids、rep_points都变为NxKx2矩阵
# rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
# rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
# sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids), 2)
# # 获取最小值的对应索引值
# best_centroids = tf.argmin(sum_squares, 1)
# # 判断所有族中心是否不再变化
# did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))
#
#
# #定义函数，更新各族中心坐标
# def bucket_mean(data, bucket_ids, num_buckets):
#     total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
#     count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
#     return total / count
# means = bucket_mean(points, best_centroids, K)
# #确定执行依赖关系，先执行did_assignments_change，然后执行后续命令
# with tf.control_dependencies([did_assignments_change]):
#     do_updates = tf.group(centroids.assign(means),cluster_assignments.assign(best_centroids))
#
# fig, ax = plt.subplots()
# colourindexes=[2,1,3]
# #循环停止条件是族中心不再变化而且循环次数不超过指定值
# while changed and iters < MAX_ITERS:
#     fig, ax = plt.subplots()
#     iters += 1
#     [changed, _] = sess.run([did_assignments_change, do_updates])
#     [centers, assignments] = sess.run([centroids, cluster_assignments])
#     ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker = 'o', s = 200, c = assignments, cmap=plt.cm.coolwarm )
#     ax.scatter(centers[:,0],centers[:,1], marker = '^', s = 550, c = colourindexes, cmap=plt.cm.plasma)
#     ax.set_title('Iteration ' + str(iters))
#     plt.savefig("kmeans" + str(iters) +".png")
# ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker = 'o', s = 200, c = assignments, cmap=plt.cm.coolwarm )
# plt.show()

# import numpy as np  # 科学计算包
# import matplotlib.pyplot as plt  # python画图包
# from matplotlib.font_manager import FontProperties
#
# from sklearn.cluster import KMeans  # 导入K-means算法包
#
# import matplotlib.font_manager as fm  ###便于中文显示
#
# # myfont = fm.FontProperties(fname='/home/hadoop/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf')
# myfont = FontProperties(fname=r"c:\windows\fonts\simkai.ttf", size=14)
#
# X=np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9],[3,4],[2,3]])
# # X=np.array([[2,1],[2,2],[5,1],[5, 2]])
# # y=np.array([0,1,1,1,1,2,2,2])
#
# plt.figure(figsize=(8, 6))
#
# '''''
# centers:产生数据的聚类中心点，默认值3
# init:采用随机还是k-means++等（可使聚类中心尽可能的远的一种方法）
# random_state:随机生成器的种子
#
# '''
# random_state = 10
#
# # 使用k-means聚类
#
# plt.subplot(1,2,1)  # 在2图里添加子图1
# y_pred = KMeans(n_clusters=3, init='random', random_state=random_state).fit_predict(X)
# print(y_pred)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.title("使用kMeas聚类", fontproperties=myfont, size=12)  # 加标题
#
# # 使用kmeans++聚类
# y_pred = KMeans(n_clusters=3, init='k-means++', random_state=random_state).fit_predict(X)
#
# plt.subplot(1,2,2)  # 在2图里添加子图2
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.title("使用kMeans++聚类", fontproperties=myfont, size=12)
#
# plt.show()

# import numpy as np  # 科学计算包
# import matplotlib.pyplot as plt  # python画图包
#
# def initkofCluster(kNums, AllNums):
#     select_list = np.arange(AllNums)
#     np.random.shuffle(select_list)
#     return select_list[0:kNums]
#
# def initkofCluster1(kNums, AllNums):
#     '''
#     随机从AllNums序列中生成前kNums
#     :param kNums: 类数
#     :param AllNums: 总数据数
#     :return: 前kNums抽样点
#     '''
#     listNum = []
#     for i in range(kNums):
#         thisknum = np.random.randint(0, AllNums)
#         while thisknum in listNum:
#             thisknum = np.random.randint(0, AllNums)
#         listNum.append(thisknum)
#     return listNum
#
# def calDistance(x1, x2):
#     '''
#     计算距离
#     :param x1: 点1
#     :param x2: 点2
#     :return: 距离
#     '''
#     return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
#
# def initkofCluster2(kNums, AllNums, clusterData):
#     '''
#     随机从AllNums序列中生成一个点，然后再计算距离较大的点作为后续点，生成前kNums
#     :param kNums: 类数
#     :param AllNums: 总数据数
#     :param clusterData: 欲聚类数据
#     :return: 前kNums抽样点
#     '''
#     listNum = []
#     select_list = np.arange(AllNums)
#     np.random.shuffle(select_list)
#     listNum.append(select_list[0])
#     for i in range(1, kNums):
#         allDist = np.zeros(AllNums - i)
#         for j in listNum:
#             thisDot = np.array(clusterData[j])
#             thisDist = [calDistance(clusterData[k], thisDot) for k in select_list[i:]]
#             allDist = allDist + thisDist
#         max_index = list(allDist).index(max(allDist))  # 返回最大值的索引
#         listNum.append(select_list[max_index + i])
#     return listNum
#
# def kMeansClassify(kNums, initkNums, clusterData):
#     '''
#     聚类并返回结果
#     :param kNums: 类数
#     :param initkNums: 前kNums抽样点
#     :param clusterData: 欲聚类数据
#     :return: 聚类结果
#     '''
#     oldCenters = [clusterData[j] for j in initkNums]
#     datanums = clusterData.shape[0]
#     centerChange = 1
#     stepTime = 0
#     while centerChange == 1 and 10 > stepTime:
#         newCenters = np.zeros(np.shape(oldCenters))
#         kMeansClass = []
#         for i in range(datanums):
#             distNums = [calDistance(clusterData[i], oldCenters[j]) for j in range(kNums)]
#             sortDist = sorted(enumerate(distNums), key=lambda x: x[1])
#             kMeansClass.append(sortDist[0][0])
#         for j in range(kNums):
#             thisCenter = []
#             for i in range(datanums):
#                 if kMeansClass[i] == j:
#                     thisCenter.append(i)
#             centerDot = np.array([clusterData[i] for i in thisCenter])
#             newCenters[j] = centerDot.sum(axis=0) / len(thisCenter)
#         centerChange = 0
#         if 0.00001 < np.max(np.array(newCenters - oldCenters)):
#             centerChange = 1
#         if centerChange == 1:
#             oldCenters = newCenters
#         stepTime += 1
#     return kMeansClass
#
# X=np.array([[2.,10.],[2.,5.],[8.,4.],[5.,8.],[7.,5.],[6.,4.],[1.,2.],[4.,9.],[3.,4.],[2.,3.],[8.,6.],[1.,4.],[7.,4.]])
# plt.subplot(1, 3, 1)
# plt.scatter(X[:, 0], X[:, 1])
# allNums = X.shape[0]
# Y=np.zeros(allNums)
# classNum = 3
# # initkNums = initkofCluster(classNum, allNums)
# initkNums = initkofCluster2(classNum, allNums, X)
# plt.subplot(1, 3, 2)
# thisDot = 0
# for i in initkNums:
#     thisDot += 1
#     Y[i] = 1
#     plt.text(X[i, 0], X[i, 1], str(thisDot))
# plt.scatter(X[:, 0], X[:, 1], c = Y)
# kMeansClass = kMeansClassify(classNum, initkNums, X)
# plt.subplot(1, 3, 3)
# plt.scatter(X[:, 0], X[:, 1], c = [i / (classNum - 1) for i in kMeansClass])
# plt.show()


import numpy as np  # 科学计算包
import matplotlib.pyplot as plt  # python画图包

def calDistance(x1, x2):
    '''
    计算距离
    :param x1: 点1
    :param x2: 点2
    :return: 距离
    '''
    # return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)

def initkofCluster(kNums, clusterData, kType = 'kMeans++'):
    '''
    随机从AllNums序列中生成一个点，然后再计算距离较大的点作为后续点，生成前kNums
    :param kNums: 类数
    :param clusterData: 欲聚类数据
    :param kType: 初始化方法
    :return: 前kNums抽样点
    '''
    AllNums = clusterData.shape[0]
    if 'kMeans++' != kType:
        select_list = np.arange(AllNums)
        np.random.shuffle(select_list)
        return select_list[0:kNums]

    listNum = []
    select_list = np.arange(AllNums)
    np.random.shuffle(select_list)
    listNum.append(select_list[0])
    for i in range(1, kNums):
        allDist = np.zeros(AllNums - i)
        for j in listNum:
            thisDot = np.array(clusterData[j])
            thisDist = [calDistance(clusterData[k], thisDot) for k in select_list[i:]]
            allDist = allDist + thisDist
        max_index = list(allDist).index(max(allDist))  # 返回最大值的索引
        listNum.append(select_list[max_index + i])
    return listNum

def kMeansClassify(kNums, initkNums, clusterData):
    '''
    聚类并返回结果
    :param kNums: 类数
    :param initkNums: 前kNums抽样点
    :param clusterData: 欲聚类数据
    :return: 聚类结果
    '''
    oldCenters = [clusterData[j] for j in initkNums]
    datanums = clusterData.shape[0]
    centerChange = 1
    stepTime = 0
    while centerChange == 1 and 10 > stepTime:
        newCenters = np.zeros(np.shape(oldCenters))
        kMeansClass = []
        for i in range(datanums):
            distNums = [calDistance(clusterData[i], oldCenters[j]) for j in range(kNums)]
            sortDist = sorted(enumerate(distNums), key=lambda x: x[1])
            kMeansClass.append(sortDist[0][0])
        for j in range(kNums):
            thisCenter = []
            for i in range(datanums):
                if kMeansClass[i] == j:
                    thisCenter.append(i)
            centerDot = np.array([clusterData[i] for i in thisCenter])
            newCenters[j] = centerDot.sum(axis=0) / len(thisCenter)
        centerChange = 0
        if 0.00001 < np.max(np.array(newCenters - oldCenters)):
            centerChange = 1
        if centerChange == 1:
            oldCenters = newCenters
        stepTime += 1
    return kMeansClass

X=np.array([[2.,10.],[2.,5.],[8.,4.],[5.,8.],[7.,5.],[6.,4.],[1.,2.],[4.,9.],[3.,4.],[2.,3.]])
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1])
allNums = X.shape[0]
Y=np.zeros(allNums)
classNum = 3
initkNums = initkofCluster(classNum, X, 'kMeans++')
plt.subplot(1, 3, 2)
thisDot = 0
print(initkNums)
for i in initkNums:
    thisDot += 1
    Y[i] = 1
    plt.text(X[i, 0], X[i, 1], str(thisDot))
plt.scatter(X[:, 0], X[:, 1], c = Y)
kMeansClass = kMeansClassify(classNum, initkNums, X)
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c = [i / (classNum - 1) for i in kMeansClass])
plt.show()