import numpy as np
import pandas as pd
import random

def data_generate1():
    data_set=[]
    for i in range(5):
        x0 = 1
        x1 = random.randint(-10, 50)
        x2 = random.randint(1, 50)
        x3 = random.randint(-10, 50)
        x4 = random.randint(1, 50)
        y = ( 5 + 0.3 * x1 + 2 * x2 + 0.5 * x3 + x4) + random.randint(-5, 5)
        data_set.append([x0, x1, x2, x3, x4, y])

    # print(data_set)
    data = pd.DataFrame(data_set, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'y'])

    return data

def data_generate2():
    data_set=[]
    for i in range(5):
        x0 = 1
        x1 = random.randint(-10, 50)
        x2 = random.randint(1, 50)
        x3 = random.randint(-10, 50)
        x4 = random.randint(1, 50)
        y = ( 5 + 0.3 * x1 + 2 * x2 + 0.5 * x3 + x4) + random.randint(-5, 5)
        data_set.append([x0, x1, x2, x3, x4, y])

    # print(data_set)
    data = pd.DataFrame(data_set)

    return data




if __name__ == "__main__":

    data=data_generate1()
    print(data,'------',type(data),'\n')

    x=data[['x1', 'x2', 'x3', 'x4']]
    print(x,"--------",type(x),'\n')# x是pandas数据框


    y1=data['y']
    print(y1,"--------",type(y1),"\n")

    y2 = data[['y']]
    print(y2, "--------", type(y1))

    theta=np.ones(x.shape[1])
    print(theta)
    # dot=np.dot(x,theta.T)
    dot1=x.dot(theta.T)
    print(dot1,'------------',type(dot1),'\n')#dot1是pandas序列
    dot2= np.dot(x,theta.T)
    print(dot2, '------------', type(dot2))#dot2是numpy数组

    # data2=data_generate2()
    # print(data2[])


