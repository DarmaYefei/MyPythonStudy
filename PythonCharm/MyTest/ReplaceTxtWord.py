# _*_ encoding=UTF-8 _*_
import re
import os
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def replaceDToFB():
    fileExsit = os.path.exists('FIBERMEDICAL.txt')
    if fileExsit == True:
        f2 = open('FIBERMEDICAL.txt', 'w')
        f2.truncate()

    f = open("DARMA.txt", "r")
    strinfo = re.compile('DARMA')
    line = 'line'
    while line:
        line = f.readline()
        newLine = strinfo.sub('FIBERMEDICAL', line)
        with open('FIBERMEDICAL.txt', 'a') as f2:
            f2.write(newLine)

    f.close()
    f2.close()

def replaceDToFBXls():
    fileExsit = os.path.exists('FIBERMEDICAL.txt')
    if fileExsit == True:
        f2 = open('FIBERMEDICAL.txt', 'w')
        f2.truncate()

    ExcelFile = xlrd.open_workbook('DataUnit.xlsx')
    sheet = ExcelFile.sheet_by_name('Sheet1')
    rowNum = sheet.nrows

    f = open("Untitled.txt", "r")
    strinfo = re.compile('DARMA')
    line = 'line'
    while line:
        line = f.readline()
        newLine = line
        for i in range(rowNum):
            ExcelRows = sheet.row_values(i)
            strinfo = re.compile(ExcelRows[0])
            newLine = strinfo.sub(ExcelRows[1], newLine)
        with open('FIBERMEDICAL.txt', 'a') as f2:
            f2.write(newLine)

    f.close()
    f2.close()

def statisticAFPrecise(filePath):
    df = pd.DataFrame(pd.read_excel(filePath))
    dfResult = df[['文件名', '混沌性']]
    nameAF = np.zeros(dfResult.shape[0])
    calYesorNo = list('?' * dfResult.shape[0])
    cutOff = 0
    SensitivityErrorN = 0
    SpecificityErrorN = 0
    for i in range(dfResult.shape[0]):
        strName = dfResult['文件名'][i]
        if 0 < strName.find('AF'):
            nameAF[i] = 1
            if cutOff > dfResult['混沌性'][i]:
                calYesorNo[i] = 'AFFalse'
                SensitivityErrorN = SensitivityErrorN + 1
                print('第%s个文件%s=%s' % (i + 1, dfResult['文件名'][i], dfResult['混沌性'][i]))
        else:
            if cutOff < dfResult['混沌性'][i]:
                calYesorNo[i] = 'NormalFalse'
                SpecificityErrorN = SpecificityErrorN + 1
                print('第%s个文件%s=%s' % (i + 1, dfResult['文件名'][i], dfResult['混沌性'][i]))

    SensitivityN = sum(nameAF == 1)
    SpecificityN = dfResult.shape[0] - SensitivityN
    calYesorNoNp = np.array(calYesorNo)

    dfResult.insert(2, 'AF是否存在', nameAF)
    dfResult.insert(3, '准确性', calYesorNoNp)

    print('总数=%s 正例数=%s 反例数=%s' % (dfResult.shape[0], SensitivityN, SpecificityN))
    print('正例错误数=%s 反例错误数=%s' % (SensitivityErrorN, SpecificityErrorN))
    print('敏感性=%s 特异性=%s 准确性=%s\n' % ((1 - SensitivityErrorN / SensitivityN), (1 - SpecificityErrorN / SpecificityN),
                                      (1 - (SensitivityErrorN + SpecificityErrorN) / dfResult.shape[0])))

def plotTest():
    x = np.arange(-5.0, 5.0, 0.01)
    y1 = np.sin(x)
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title('sin(x)', color=(0.5, 0.8, 0.3))
    plt.plot(x, y1, 'b')
    plt.grid(True)
    plt.annotate('local zeros', xy=(0, 0), xytext=(1, -0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.axis([-5, 5, -2, 2])
    plt.subplot(2, 1, 2)
    # 设置x轴范围
    plt.xlim(-2.5, 2.5)
    # 设置y轴范围
    plt.ylim(-1, 1)
    plt.plot(x, y1, color="red", linewidth=1, linestyle="-.", label="sine")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('Time[sec]')
    plt.text(1, 0, r'$\mu=100,\ \sigma=15$')
    plt.show()

if __name__ == '__main__':
    replaceDToFBXls()
    # statisticAFPrecise('ResultWH.xls')
    # statisticAFPrecise('StatisticResult20181113150304.xls')
    # plotTest()





