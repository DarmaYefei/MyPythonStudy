import numpy as np
import xlrd

filePath = 'pythonAFdetect.xlsx'
book = xlrd.open_workbook(filePath)
sheet = book.sheet_by_index(0)

data = np.zeros((sheet.nrows, sheet.ncols))

for i in range(sheet.nrows):
    data[i, :] = sheet.row_values(i)

x = data[:, 0:300]
y = data[:, 300]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.20)

from sklearn import svm
clf = svm.SVC(gamma=0.1, C=1.5, probability=True)
clf.fit(Xtrain, Ytrain)

from sklearn.metrics import accuracy_score

print(Ytest)
Ypred = clf.predict(Xtest)
accuracy_score(Ytest, Ypred)
print(Ypred-Ytest)
print(clf.score(Xtest, Ytest))