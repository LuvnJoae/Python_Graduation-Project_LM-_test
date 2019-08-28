# 良品 数据周期化处理，第一步
import numpy as np
import pandas as pd
import os
import scipy.signal as ss

# 思路
    # 由于按照周期统计，良品与不良品的时间周期是不同的，而按照神经网络建立模型的话，需要指定特征，在这里，时间的幅值即特征值，有多少个离散的
    # 的时间，就有多少个特征，而在建立模型时，二者的特征数量需要一致，所以，单纯根据时间周期，是没法合理的建立特征集的。
    # 所以，第二方案，采用固定时间周期，
        # 即 从第一个峰值点开始计数，统计共1000个离散时间点的幅值，以1000作为特征集的大小。（总时间间隔周期是2000，取其一半）
        # 即使是不良，其第一个峰值点也是在1000以内的，所以完全OK
            # 对于有周期的不良数据，可以看作是周期性特征
            # 对于无明显周期的不良数据，则以1000为特征集大小，也可以保证特征集大小一致，而在训练的过程中，给无周期的加以标签，作为其他样本。

# 读取数据，只读取两列（因为用tab当作分隔符，会读出三列，但最后一列是没有的），用第一列时间作为索引,实际数据只有一列
data = pd.read_csv('../data/demo1_input/P1-01-0000.txt', sep='\t', index_col=u'Time - Plot 0', header=0, usecols=[0,1])
data = data.values #将dataFrame中 的数据转换成数组
result_matrix = np.zeros([6,1001], float) #创建一个6行，1000列的二维数组（矩阵）


max_index, max_value = ss.find_peaks(data[:, 0], height= 0.06, distance=200)# 设置每个周期限制，高度为0.06，距离为200  max_index为最高点下标，max_value为最高点的值
time_index = max_index * 0.000005 # max_index是从1开始的下标,所以乘上一个时间间隔，得到实际时间下标

temp = data[max_index[0]: max_index[0]+1000].T # 从第一个峰值开始，往后数1000个数据，将这些数据给result_matrix矩阵中
result_matrix[0, :temp.size] = temp
result_matrix[0, 1000] = 0 #写标签

print(result_matrix)


