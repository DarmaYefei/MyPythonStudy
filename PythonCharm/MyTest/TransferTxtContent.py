import re
import os
import xlrd


def replaceDToFBXls(OriginFile, TransToFile, DictFile):
    '''
    转换源文件TXT中的词汇
    :param OriginFile: 原始文件名（或路径）
    :param TransToFile: 存储文件名（或路径）
    :param DictFile: 字典文件名（或路径）
    :return:
    '''
    fileExsit = os.path.exists(TransToFile)
    if fileExsit == True:
        f2 = open(TransToFile, 'w')
        f2.truncate()

    ExcelFile = xlrd.open_workbook(DictFile)
    sheet = ExcelFile.sheet_by_name('Sheet1')
    rowNum = sheet.nrows

    f = open(OriginFile, "r")
    strinfo = re.compile('Love')
    line = 'line'
    while line:
        line = f.readline()
        newLine = line
        for i in range(rowNum):
            ExcelRows = sheet.row_values(i)
            strinfo = re.compile(ExcelRows[0])
            newLine = strinfo.sub(ExcelRows[1], newLine)
        with open(TransToFile, 'a') as f2:
            f2.write(newLine)

    f.close()
    f2.close()


if __name__ == '__main__':
    OriginFile = r"D:\MyPythonDoc\MyPythonStudy\MyPythonStudy\PythonCharm\MyTest\T51征信.txt"
    TransToFile = r"D:\MyPythonDoc\MyPythonStudy\MyPythonStudy\PythonCharm\MyTest\Transfer51征信.txt"
    DictFile = r"D:\MyPythonDoc\MyPythonStudy\MyPythonStudy\PythonCharm\MyTest\DataUniwwt.xlsx"
    replaceDToFBXls(OriginFile, TransToFile, DictFile)

