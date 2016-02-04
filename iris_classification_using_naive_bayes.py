#! /usr/bin/env python
#coding=utf-8

# Authors: Hanxiaoyang <hanxiaoyang.ml@gmail.com>
# simple naive bayes classifier to classify iris dataset

# 代码功能：简易朴素贝叶斯分类器，直接取iris数据集，根据花的多种数据特征，判定是什么花
# 详细说明参见博客http://blog.csdn.net/han_xiaoyang/article/details/50629608
# 作者：寒小阳<hanxiaoyang.ml@gmail.com>

from sklearn import datasets
iris = datasets.load_iris()
iris.data[:5]
#array([[ 5.1,  3.5,  1.4,  0.2],
#       [ 4.9,  3. ,  1.4,  0.2],
#       [ 4.7,  3.2,  1.3,  0.2],
#       [ 4.6,  3.1,  1.5,  0.2],
#       [ 5. ,  3.6,  1.4,  0.2]])

#我们假定sepal length, sepal width, petal length, petal width 4个量独立且服从高斯分布，用贝叶斯分类器建模
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
right_num = (iris.target == y_pred).sum()
print("Total testing num :%d , naive bayes accuracy :%f" %(iris.data.shape[0], float(right_num)/iris.data.shape[0]))
# Total testing num :150 , naive bayes accuracy :0.960000
