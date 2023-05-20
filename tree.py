# -- coding: utf-8 --

import turtle as tl
import random as rd

#turtle画树
def tree(n, l):

    # l是初始前进长度  n是初始树枝粗细
    tl.pd()  # 下笔
    tl.pencolor("black") #画笔颜色

    tl.pensize(n/2)# 画笔粗细  逐渐变细
    tl.forward(l)  # 画树枝

    if n > 0:
        b = rd.random() * 10 + 10  # 右分支偏转角度   10-35
        c = rd.random() * 10 + 10  # 左分支偏转角度
        d = l * (rd.random() * 0.25 + 0.7)  # 下一个分支的长度        [0.7,0.95)前进距离随机取

        # 右转一定角度,画右分支
        tl.right(b)
        tree(n - 1, d)
        # 左转一定角度，画左分支
        tl.left(b + c)
        tree(n - 1, d)

        # 转回来
        tl.right(c)
    else:
        # 画叶子
        tl.right(90)

        ran = rd.random()
        # 这里相比于原来随机添加了填充的圆圈，让樱花叶子看起来更多一点
        if (ran > 0.5):
            tl.begin_fill()
            tl.circle(5)# 默认逆时针画圈
            r = rd.random()
            g = rd.random()
            b = rd.random()
            tl.fillcolor(r,g,b)

        # 把原来随机生成的叶子换成了统一的粉色
        r = rd.random()
        g = rd.random()
        b = rd.random()
        tl.pencolor(r,g,b) #   画一个粉色轮廓的叶子
        tl.circle(5)

        if (ran > 0.5):
            tl.end_fill()# 完成叶子颜色填充

        tl.left(90)

        # 添加0.3倍的飘落叶子
        if (rd.random() > 0.7):
            tl.pu()#抬笔
            # 飘落
            t = tl.heading()  #当前航向
            an = -20+ rd.random() * (-140) #  -2--(-160)
            tl.setheading(an) #每次setheading(to_angle) 小海龟以正东（X轴正方向）为基准转向to_angle角度  to_angle为 正 逆时针转向 ////   to_angle为 负 顺时针转向
            dis = int(rd.random()*400)# 跳跃多少距离
            tl.forward(dis)

            # 画叶子
            tl.pd()
            r = rd.random()
            g = rd.random()
            b = rd.random()
            tl.pencolor(r, g, b)  # 取【0，1】范围小数
            tl.circle(5)


            # 返回
            tl.pu()
            tl.setheading(an)
            tl.backward(dis)
            tl.setheading(t)

    tl.pu()#抬笔
    tl.backward(l)  # 退回


def turtle_init():

    # turtle箭头初始默认朝右---》
    tl.bgcolor(1, 0.92, 0.80)  # 设置面板背景色
    # ht()  # 隐藏turtle
    # speed(0)  # 速度 1-10渐进，0 最快
    tl.tracer(0, 0)  # 动画开关  设置0为关  默认开启

    # 调整方向
    tl.pu()  # 抬笔
    tl.backward(50)
    tl.left(90)  # 左转90度
    tl.pu()  # 抬笔
    tl.backward(300)  # 后退300



if __name__=="__main__":

    turtle_init()
    tree(9,80)
    tl.done()

