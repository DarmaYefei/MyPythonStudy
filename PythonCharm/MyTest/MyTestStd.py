# from datetime import datetime, date, time
# import math
#
# print(datetime.now())
#
# today = datetime.now()
#
# print(datetime.date(today))
# print(datetime.time(today))
# print(datetime.ctime(today))
#
# date1 = date(2008, 10, 3)
# print(date1)
#
# time1 = time(20, 10, 3)
# print(time1)
#
# print(today.strftime("%Y-%m-%d %H:%M:%S %p"))
#
# print(math.trunc(3.1), end="\t")
# print(math.ceil(3.1), end="\t")
# print(math.floor(3.1), end="\t")
# print(round(3.1), end="\t")
# print("")
# print(math.trunc(3.6), end="\t")
# print(math.ceil(3.6), end="\t")
# print(math.floor(3.6), end="\t")
# print(round(3.6), end="\t")

import tkinter
root = tkinter.Tk()
root.title("hello world")
root.geometry('300x600')
# 进入消息循环
li = ['C', 'python', 'php', 'html', 'SQL', 'java']
movie = ['CSS', 'jQuery', 'Bootstrap']
listb = tkinter.Listbox(root)  # 创建两个列表组件
listb2 = tkinter.Listbox(root)
for item in li:  # 第一个小部件插入数据
    listb.insert(0, item)

for item in movie:  # 第二个小部件插入数据
    listb2.insert(0, item)

listb.pack()  # 将小部件放置到主窗口中
listb2.pack()
root.mainloop()  # 进入消息循环
