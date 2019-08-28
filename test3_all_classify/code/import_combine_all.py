
''' 将各个处理好的数据 连在一起 '''

import numpy as np
import pandas as pd
import os
import scipy.signal as ss

''' 注意：
        B8的数据，特征化太小了，所以不用这个故障类型，其他的标签依次往前调
        如： B9, B10, B11 标签分别为 7, 8 ,9
 '''
data_0 = np.loadtxt('../data/output_0/output.csv', delimiter=",")  # 读取 良品数据 数组（读得数据为数组形式）
data_B1 = np.loadtxt('../data/output_B1/output.csv', delimiter=",")  # 读取 不良品B1数据 数组（读得数据为数组形式）
data_B2 = np.loadtxt('../data/output_B2/output.csv', delimiter=",")  # 读取 不良品B2数据 数组（读得数据为数组形式）
data_B3 = np.loadtxt('../data/output_B3/output.csv', delimiter=",")  # 读取 不良品B3数据 数组（读得数据为数组形式）
data_B4 = np.loadtxt('../data/output_B4/output.csv', delimiter=",")  # 读取 不良品B4数据 数组（读得数据为数组形式）
data_B5 = np.loadtxt('../data/output_B5/output.csv', delimiter=",")  # 读取 不良品B5数据 数组（读得数据为数组形式）
data_B6 = np.loadtxt('../data/output_B6/output.csv', delimiter=",")  # 读取 不良品B6数据 数组（读得数据为数组形式）

# data_B8 = np.loadtxt('../data/output_B8/output.csv', delimiter=",")  # 读取 不良品B8数据 数组（读得数据为数组形式）
data_B9 = np.loadtxt('../data/output_B9/output.csv', delimiter=",")  # 读取 不良品B9数据 数组（读得数据为数组形式）
data_B10 = np.loadtxt('../data/output_B10/output.csv', delimiter=",")  # 读取 不良品B10数据 数组（读得数据为数组形式）
data_B11_1 = np.loadtxt('../data/output_B11_1/output.csv', delimiter=",")  # 读取 不良品B11_1数据 数组（读得数据为数组形式）
data_B11_2 = np.loadtxt('../data/output_B11_2/output.csv', delimiter=",")  # 读取 不良品B11_2数据 数组（读得数据为数组形式）
data_B11_3 = np.loadtxt('../data/output_B11_3/output.csv', delimiter=",")  # 读取 不良品B11_3数据 数组（读得数据为数组形式）

data = np.vstack((data_0, data_B1, data_B2, data_B3, data_B4, data_B5, data_B6, data_B9))

np.savetxt('../data/output/output_all.csv', data, delimiter=",")  # 将数组存储到output中
b = np.loadtxt('../data/output/output_all.csv', delimiter=",")  # 读取数组（读得数据为数组形式）
print(b)
