# -- coding: utf-8 --
# @time : 2022/11/27 
# @author : wxl
# @file : self_svc_sklearn.py
# @software: pycharm


from sklearn.datasets import load_breast_cancer# 乳腺癌数据集
from sklearn.svm import SVC #支持向量机
from sklearn.model_selection import train_test_split  #划分数据集
from sklearn.metrics import classification_report  #打印报告
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn import metrics #混淆矩阵
from  sklearn.model_selection import GridSearchCV #网格搜索 最优参数

import matplotlib.pyplot as plt
import seaborn as sns
import  pandas as pd
import numpy as np

#混淆矩阵热力图
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

#逻辑回归
def sklearn_logistic():
    data = load_breast_cancer() #加载数据集
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1) #划分数据集
    model = LogisticRegression(max_iter=3000)
    model.fit(x_train, y_train) #训练模型
    y_pred1=model.predict(x_test)
    print("sklearn_logistic模型报告:\n",classification_report(y_test, y_pred1)) #打印模型测试报告
    heatmap_generate(y_test,y_pred1,name="sklearn_logistic",color="Wistia_r")

#支持向量机
def sklearn_svc():

    # data=load_breast_cancer() #加载数据集
    # x=data.data
    # y=data.target
    path = "data/客户信息及违约表现.xlsx"
    data = pd.read_excel(path)
    x = data[['收入', '年龄', '性别', '历史授信额度', '历史违约次数']]  # 特征
    # 均值归一化
    x= (x - x.mean()) / x.std()

    y = data[["是否违约"]]  # 标签  真实值


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=1)  # 划分训练集和测试集
    model=SVC(kernel="poly",cache_size=1000,gamma="auto",degree=1) #选择模型并且设置参数
    model.fit(x_train,y_train) #训练模型
    y_pred1=model.predict(x_test)

    print("sklearn_svc模型报告:\n", classification_report(y_test, y_pred1))
    heatmap_generate(y_test, y_pred1, name="sklearn_svc", color="summer_r")

#网格法寻找最优参数
def find_parameters():

    # params={ 'C':[1,5,10,50,100,300],'kernel':["linear",'poly','rbf',"sigmoid"],\
    #          'gamma':[0.001,0.01,0.1], \
    # }
    params = {'C': [1, 5,10], 'kernel': ["linear", 'poly'] \
              }
    model=SVC(cache_size=1000)
    grid_search=GridSearchCV(model,params,cv=6)
    data = load_breast_cancer()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)  # 划分训练集和测试集
    grid_search.fit(x_train,y_train)

    print("模型最高分:{:.3f}".format(grid_search.score(x_test,y_test)))
    print("最佳参数设置:{}".format(grid_search.best_params_))



if __name__ == '__main__':

    sklearn_svc()
    sklearn_logistic()
    # find_parameters()




