#-*- coding: utf-8 -*-


from random import shuffle
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder  # 标签编码
from keras.models import Sequential  # 神经网络初始化函数
from keras.layers.core import Dense, Activation  # 神经网络层函数、激活函数

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

datafile = '../data/output/output_all.csv'
data = pd.read_csv(datafile)  # 导入数据文件
data = data.values  # 转为数组

'''对类别进行编码'''
def encodeY(y):
    encoder = LabelEncoder()  # 导入编码器
    encoder.fit(y)  # 检查需要编码的共有多少个值
    encoded_Y = encoder.transform(y)  # 将字符串值转为数值，数值编码
    dummy_y = np_utils.to_categorical(encoded_Y)  # 将数值转为二进制向量，热编码
    return dummy_y  # 返回编码后的标签向量

''' 将数据转换成含二进制编码标签的数据 '''
data_x_encode = data[:, :1000]
data_y_encode = encodeY(data[:, 1000])
data_encode = np.concatenate((data_x_encode, data_y_encode), axis=1)

np.savetxt('../data/test/output.csv', data_encode, delimiter=",")  # 将数组存储到output中

shuffle(data_encode)  # 打乱数据顺序

''' 设置训练集、验证集、测试集 '''
p = 0.8  # 设置训练数据比例,分别为训练集0.8，验证集0.1，测试集0.1

train = data_encode[:int(len(data_encode)*p), :]  # 训练集
x_tr, y_tr = train[:, :1000], train[:, 1000:1008]  # 编码

dev = data_encode[int(len(data_encode)*p): int(len(data_encode)*(p+0.2)), :]  # 验证集
x_d, y_d = dev[:, :1000], dev[:, 1000:1008]

# test = data_encode[int(len(data_encode)*(p+0.1)):, :]  # 测试集
# x_te, y_te = test[:, :1000], test[:, 1000:1012]


'''构建LM神经网络模型'''

modelfile = '../tmp/model.h5' #构建的神经网络模型存储路径

'''建立神经网络   确定每层参数与激活函数'''
model = Sequential()  # 建立神经网络模型
model.add(Dense(input_dim=1000, output_dim=30))  # 添加输入层（1000节点）到隐藏层1（10节点）的连接
model.add(Activation('relu'))  # 隐藏层使用relu激活函数
model.add(Dense(input_dim=30, output_dim=20))  # 添加隐藏层1（1000节点）到隐藏层2（10节点）的连接
model.add(Activation('relu'))  # 隐藏层使用relu激活函数
model.add(Dense(input_dim=20, output_dim=8))  # 添加隐藏层2（10节点）到输出层（12节点）的连接
model.add(Activation('softmax'))  # 输出层使用sigmoid激活函数

'''编译神经网络模型    loss是损失函数，optimizer是优化器，metrics是度量'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型，使用adam方法求解

'''训练神经网络    train是训练集  dev是验证集
        当训练完成后，model.fit会返回一个history，其中保存了训练过程中的所有数据
           有： [u'acc', u'loss', u'valacc', u'valloss'] （字典形式）
           其中，前两个是 训练集 的 准确率 和 损失函数， 后两个是 验证集 的 准确率 和 损失函数。'''
# fit参数，第一个是输入，第二个输出，第三个迭代次数，第四个批次大小，第五个验证集
history = model.fit(x_tr, y_tr, epochs=5000, batch_size=512, validation_data=(x_d, y_d))  # 训练模型，循环1000次 这里原来的nb_epoch改为epochs，

# net.save_weights(netfile)  # 仅保存模型的权重参数，导入前需要规定好神经网络的结构，导入用model.load_weight，占空间较小
model.save(modelfile)  # 保存模型整体结构，导入用 model.load() 占空间最大，保存的是.h5文件

'''预测结果     这里仅展示predict_classes的用法，实际绘制ROC曲线图 与 混淆矩阵时，要通过test，即测试集的数据，来评判性能指标。
    ## 注： keras中，
        predict：表示预测概率，
        predict_classes：表示预测类别
        两者的预测结果都是n * 1维数组，而不是通常的 1 * n'''
# predict_result = model.predict_classes(train[:,:1000]).reshape(len(train)) #训练集，预测结果
# predict_result = model.predict_classes(dev[:,:1000]).reshape(len(dev)) #验证集，预测结果
predict_result = model.predict_classes(x_tr).reshape(len(train)) #测试集，预测结果（通过测试集来进行性能评判）

print(predict_result)


'''性能分析  loss曲线'''
def loss(history):
    import matplotlib.pyplot as plt

    ''' 当训练完成后，model.fit会返回一个history，其中保存了训练过程中的所有数据
            有： [u'acc', u'loss', u'valacc', u'valloss'] （字典形式）
            其中，前两个是 训练集 的 准确率 和 损失函数， 后两个是 验证集 的 准确率 和 损失函数。
        '''
    history_dict = history.history
    loss_values = history_dict['loss']  # 取出训练集的loss值
    val_loss_values = history_dict['val_loss']  # 取出验证集的loss值
    epochs = range(1, len(loss_values)+1)  # 设 迭代次数 为 损失函数 的个数，即全部的迭代次数，间隔1
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validattion loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

loss(history)

'''性能分析  acc曲线'''
def acc(history):
    import matplotlib.pyplot as plt

    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, 'bo',label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validattion acc')
    plt.title('Training and Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('accuary')
    plt.legend()
    plt.show()

acc(history)


# '''性能评判  混淆矩阵'''
# from cm_plot import *  # 导入自行编写的混淆矩阵可视化函数
# # cm_plot(train[:, 1000], predict_result).show() # 显示 训练集 混淆矩阵可视化结果
# cm_plot(y_te, predict_result).show()  # 显示 训练集 混淆矩阵可视化结果

# '''性能评判  ROC曲线'''
# from sklearn.metrics import roc_curve  # 导入ROC曲线函数
# import matplotlib.pyplot as plt
#
# fpr, tpr, thresholds = roc_curve(y_te, predict_result, pos_label=1)
# plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')  # 作出ROC曲线
# plt.xlabel('False Positive Rate')  # 坐标轴标签
# plt.ylabel('True Positive Rate')  # 坐标轴标签
# plt.ylim(0, 1.05)  # 边界范围
# plt.xlim(0, 1.05)  # 边界范围
# plt.legend(loc=4)  # 图例
# plt.show()  # 显示作图结果