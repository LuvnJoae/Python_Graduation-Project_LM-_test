
''' 导入模型，进行手动测试 '''

import pandas as pd
from keras.models import load_model
from keras.models import Sequential #导入神经网络初始化函数

''' 测试 ： 0， B1~B6'''
for i in range(0, 7):
    datafile = '../data/test/B' + str(i) + '/output.csv'
    data = pd.read_csv(datafile)
    data = data.values

    modelfile = '../tmp/model.h5'

    net = load_model(modelfile)

    predict_result = net.predict_classes(data[:, :1000]).reshape(len(data)) #预测结果变形
    '''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''

    print(predict_result)

''' 测试 ： B9~B10'''
for i in range(9, 11):
    datafile = '../data/test/B' + str(i) + '/output.csv'
    data = pd.read_csv(datafile)
    data = data.values

    modelfile = '../tmp/model.h5'

    net = load_model(modelfile)

    predict_result = net.predict_classes(data[:, :1000]).reshape(len(data)) #预测结果变形
    '''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''

    print(predict_result)

# ''' 测试 ： B11_1~B11_3'''
# for i in range(1, 4):
#     datafile = '../data/test/B11_' + str(i) + '/output.csv'
#     data = pd.read_csv(datafile)
#     data = data.values
#
#     modelfile = '../tmp/model.h5'
#
#     net = load_model(modelfile)
#
#     predict_result = net.predict_classes(data[:, :1000]).reshape(len(data)) #预测结果变形
#     '''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''
#
#     print(predict_result)

# datafile = '../data/test/B11_1/output.csv'
# data = pd.read_csv(datafile)
# data = data.values
#
# modelfile = '../tmp/model.h5'
#
# net = load_model(modelfile)
#
# predict_result = net.predict_classes(data[:, :1000]).reshape(len(data)) #预测结果变形
# '''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''
#
# print(predict_result)