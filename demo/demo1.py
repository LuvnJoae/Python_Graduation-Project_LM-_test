# 数据周期化处理，第一步
import numpy as np
import pandas as pd
import os
import scipy.signal as ss


# 读取数据，只读取两列（因为用tab当作分隔符，会读出三列，但最后一列是没有的），用第一列时间作为索引,实际数据只有一列
data = pd.read_csv('../test1/data/P1-01-0000.txt', sep='\t', index_col=u'Time - Plot 0', header=0, usecols=[0,1])
data = data.values #将dataFrame中 的数据转换成数组
result_matrix = np.zeros([6,304], float) #创建一个6行，304列的二维数组（矩阵）(304是所有数据统计下来的单一周期的最大值)

max_index, max_value = ss.find_peaks(data[:, 0], height= 0.06, distance=200)# 设置每个周期限制，高度为0.06，距离为200
    # max_index为最高点下标，max_value为最高点的值

print(max_index)