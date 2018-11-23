from sklearn import svm

if __name__ == '__main__':
    x = [[2, 0, 1], [1, 1, 2], [2, 3, 3]]
    y = [0, 0, 1]  # 分类标记
    clf = svm.SVC(kernel='linear')  # SVM模块，svc,线性核函数
    clf.fit(x, y)
    print(clf)
    print(clf.support_vectors_)
    print(clf.support_)
    print(clf.n_support_)
    #print(clf.predict([2, 0, 3]))