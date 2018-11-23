import tkinter as tk
import random as rd

if __name__ == "__main__":
    sysNumber = 0
    root = tk.Tk()
    root.title("猜数字")
    root.geometry("400x300")
    # 顶部提示
    label = tk.Label(root, text='你好，欢迎来玩猜数字游戏！')  # 生成标签
    label.pack()

    var = tk.StringVar()  # 文字变量储存器
    var.set('')
    label2 = tk.Label(root,
                 textvariable=var,  # 使用 textvariable 替换 text, 因为这个是可以变化的
                 bg='green', width=30, height=2)
    label2.pack()

    def hit_me():
        global sysNumber
        sysNumber = rd.randrange(1, 1000)
        var.set('你已点击生成数字：请猜猜为？')

    # 按钮生成
    button1 = tk.Button(root, text='点击生成随机数字[1~1000]', command=hit_me)  # 生成button1
    button1.pack()  # 将button1添加到root主窗口

    t = tk.Entry(root, width=30)  # 创建文本框，用户可输入内容
    t.pack()

    var1 = tk.StringVar()  # 文字变量储存器
    var1.set('')
    def hit_me1():
        global sysNumber
        guessNumber = t.get()
        if not guessNumber.isdigit():
            var1.set('请确认输入的只能为数字！')
        else:
            if int(guessNumber) < sysNumber:
                var1.set('你所猜的数字为：' + guessNumber + '很遗憾小了')
            elif int(guessNumber) > sysNumber:
                var1.set('你所猜的数字为：' + guessNumber + '很遗憾大了')
            else:
                var1.set('你所猜的数字为：' + guessNumber + '恭喜你正确！')
                var.set('你所生成的数字猜为 %d' % sysNumber)

    button2 = tk.Button(root, text='猜定数字', command=hit_me1)  # 生成button1
    button2.pack()  # 将button1添加到root主窗口

    label3 = tk.Label(root,
                      textvariable=var1,  # 使用 textvariable 替换 text, 因为这个是可以变化的
                      bg='DarkOrchid1', width=30, height=2)
    label3.pack()
    root.mainloop()
