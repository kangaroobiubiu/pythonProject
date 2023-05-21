# -- coding: utf-8 --
# @time : 2022/11/21 
# @author : wxl
# @file : self_sklearn_logistic.py
# @software: pycharm



from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


#鸾尾花集 分类

def data_normalization(X):  #数据归一化
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def dataLoad():
    data = load_iris() #加载鸾尾花集
    x, y = data.data, data.target.reshape(-1, 1)
    x = data_normalization(x) #数据归一化
    return x, y


def accuracy(y, y_pre):
    return np.mean((y.flatten() == y_pre.flatten()) * 1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def hypothesis(X, W, bias):
    z = np.matmul(X, W) + bias
    h_x = sigmoid(z)
    return h_x


def prediction(X, W, bias):
    class_type = len(W) # 多种预测系数列表
    prob = []
    for c in range(class_type):
        w, b = W[c], bias[c]
        h_x = hypothesis(X, w, b)
        prob.append(h_x)
    prob = np.hstack(prob)  # 水平堆叠前是3个 150*1           水平堆叠后变成 一个 150*3
    y_pre = np.argmax(prob, axis=1)  #  比较行 返回列 最大 的索引
    return y_pre


def cost_function(X, y, W, bias):
    m, n = X.shape
    h_x = hypothesis(X, W, bias) #预测值
    loss= -(y * np.log(h_x) + (1 - y) * np.log(1 - h_x)) #损失函数
    cost = np.sum(loss)
    return cost / m


def GradientAscent(X, y, W, bias, alpha):
    m, n = X.shape
    h_x = hypothesis(X, W, bias)
    grad_w = (1 / m) * np.matmul(X.T, (h_x - y))  # [n,m] @ [m,1]
    grad_b = (1 / m) * np.sum(h_x - y)
    W = W - alpha * grad_w  # 梯度下降
    bias = bias - alpha * grad_b
    return W, bias


def train_binary(X, y, iter=200):
    m, n = X.shape  # 506,13
    W = np.random.randn(n, 1)  # 0.953
    b, alpha, costs = 0.3, 0.5, []
    for i in range(iter):
        costs.append(cost_function(X, y, W, b))
        W, b = GradientAscent(X, y, W, b, alpha)
    return costs, W, b


def train(x, y, iter=1000):
    class_type = np.unique(y) # 标签 种类 数量
    costs, W, b = [], [], []
    for c in class_type:
        label = (y == c) * 1
        tmp = train_binary(x, label, iter=iter)
        costs.append(tmp[0])
        W.append(tmp[1])
        b.append(tmp[2])
    costs = np.vstack(costs)   #3*1000
    costs = np.sum(costs, axis=0) # 列数保持不变   每一列上求和
    y_pre = prediction(x, W, b)
    print("个人模型:",classification_report(y, y_pre))
    print('个人模型准确率=%.2f'%(accuracy(y, y_pre)))
    return costs,y_pre

def heatmap_generate(Y,y_pred,color="Wistia_r",name='selfModel Confusion matrix'):

    cnf_matrix = metrics.confusion_matrix(Y, y_pred)  # 生成混淆矩阵
    # print(cnf_matrix)

    class_names = [0, 1]  # 下标刻度 名字
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names)) # 下标刻度 范围
    plt.xticks(tick_marks, class_names)  #修改x轴下标刻度范围以及名字
    plt.yticks(tick_marks, class_names)
    # 创建热力图
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap=color, fmt='g')
    ax.xaxis.set_label_position("top")  # x轴 y轴 说明标签的位置
    # plt.tight_layout()# 调整 子图 适应整个fig  实验功能 不稳定
    plt.title(name, y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def sklearnModel(x, y):
    model = LogisticRegression()
    model.fit(x, y)
    y_pred1=model.predict(x)
    print("sklearn模型:",classification_report(y, y_pred1))
    print("sklearn模型准确率=%.2f"%model.score(x, y))
    heatmap_generate(y,y_pred1,name="sklearnModel",color="summer_r")


if __name__ == '__main__':
    x, y = dataLoad()
    costs,y_pred= train(x, y)#个人模型
    heatmap_generate(y,y_pred)

    plt.xlabel("iterations")  # X轴含义
    plt.ylabel("cost")  # y轴含义
    x_scale = range(len(costs))
    plt.plot(x_scale, costs, label="selfModel", color='red')
    plt.legend()
    plt.show()


    sklearnModel(x, y)#sklearn模型




