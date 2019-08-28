#-*- coding: utf-8 -*-

import pandas as pd

datafile = '../data/output_0&B1_All/output_test_B1.csv'
data = pd.read_csv(datafile)
data = data.values

from keras.models import load_model
from keras.models import Sequential #导入神经网络初始化函数

modelfile = '../tmp/model.h5'

net = load_model(modelfile)

predict_result = net.predict_classes(data[:,:1000]).reshape(len(data)) #预测结果变形
'''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''

print(predict_result)