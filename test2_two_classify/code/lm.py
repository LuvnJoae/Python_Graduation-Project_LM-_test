#-*- coding: utf-8 -*-

import pandas as pd
from random import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# datafile = '../data/output_0many&B1/output.xlsx'
datafile = '../data/output_0&B1_All/output.csv'
data = pd.read_csv(datafile)
data = data.values
shuffle(data)

print(data)

p = 0.8 #设置训练数据比例,分别为训练集0.8，验证集0.1，测试集0.1
train = data[:int(len(data)*p), :]  # 训练集
dev = data[int(len(data)*p): int(len(data)*(p+0.1)), :] # 验证集
test = data[int(len(data)*(p+0.1)):, :] # 测试集

#构建LM神经网络模型[[
from keras.models import Sequential #导入神经网络初始化函数
from keras.layers.core import Dense, Activation #导入神经网络层函数、激活函数

modelfile = '../tmp/model.h5' #构建的神经网络模型存储路径

'''建立神经网络   确定每层参数与激活函数'''
model = Sequential()  # 建立神经网络模型
model.add(Dense(input_dim=1000, output_dim=10))  # 添加输入层（1000节点）到隐藏层（10节点）的连接
model.add(Activation('relu'))  # 隐藏层使用relu激活函数
model.add(Dense(input_dim=10, output_dim=1))  # 添加隐藏层（10节点）到输出层（1节点）的连接
model.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数

'''编译神经网络模型    loss是损失函数，optimizer是优化器，metrics是度量'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型，使用adam方法求解

'''训练神经网络    train是训练集  dev是验证集'''
# train里的参数，第一个是输入，第二个输出，第三个迭代次数，第四个分组
history = model.fit(train[:, :1000], train[:, 1000], batch_size=100, nb_epoch=1000, validation_data=(dev[:, :1000], dev[:, 1000]))  # 训练模型，循环1000次

# net.save_weights(netfile)  # 仅保存模型的权重参数，导入前需要规定好神经网络的结构，导入用model.load_weight，占空间较小
model.save(modelfile)  # 保存模型整体结构，导入用 model.load() 占空间最大，保存的是.h5文件

'''预测结果     这里仅展示predict_classes的用法，实际绘制ROC曲线图 与 混淆矩阵时，要通过test，即测试集的数据，来评判性能指标。
    ## 注： keras中，
        predict：表示预测概率，
        predict_classes：表示预测类别
        两者的预测结果都是n * 1维数组，而不是通常的 1 * n'''
# predict_result = model.predict_classes(train[:,:1000]).reshape(len(train)) #训练集，预测结果
# predict_result = model.predict_classes(dev[:,:1000]).reshape(len(dev)) #验证集，预测结果
predict_result = model.predict_classes(test[:, :1000]).reshape(len(test)) #测试集，预测结果（通过测试集来进行性能评判）

'''性能评判  混淆矩阵'''
from cm_plot import *  # 导入自行编写的混淆矩阵可视化函数
# cm_plot(train[:, 1000], predict_result).show() # 显示 训练集 混淆矩阵可视化结果
cm_plot(test[:, 1000], predict_result).show()  # 显示 训练集 混淆矩阵可视化结果

'''性能评判  ROC曲线'''
from sklearn.metrics import roc_curve  # 导入ROC曲线函数
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(test[:, 1000], predict_result, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')  # 作出ROC曲线
plt.xlabel('False Positive Rate')  # 坐标轴标签
plt.ylabel('True Positive Rate')  # 坐标轴标签
plt.ylim(0, 1.05)  # 边界范围
plt.xlim(0, 1.05)  # 边界范围
plt.legend(loc=4)  # 图例
plt.show()  # 显示作图结果

'''性能分析  loss曲线'''
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

'''性能分析  acc曲线'''
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