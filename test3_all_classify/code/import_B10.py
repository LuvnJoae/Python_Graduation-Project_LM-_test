''' 思路
    # 由于按照周期统计，良品与不良品的时间周期是不同的，而按照神经网络建立模型的话，需要指定特征，在这里，时间的幅值即特征值，有多少个离散的
    # 的时间，就有多少个特征，而在建立模型时，二者的特征数量需要一致，所以，单纯根据时间周期，是没法合理的建立特征集的。
    # 所以，第二方案，采用固定时间周期，
        # 即 从第一个峰值点开始计数，统计共1000个离散时间点的幅值，以1000作为特征集的大小。（总时间间隔周期是2000，取其一半）
        # 即使是不良，其第一个峰值点也是在1000以内的，所以完全OK
            # 对于有周期的不良数据，可以看作是周期性特征 如： B1,3,4,6,8,9,10,11
            # 对于无明显周期的不良数据，则以1000为特征集大小，也可以保证特征集大小一致。如：B2,5'''

''' 数据预处理  生成标准CSV格式表格数据 ，含1000个特征，一个未编码标签 

    不良品 B10 数据，标签为 8 （缺少B7，去除B8，所以标签前移）
    '''
import numpy as np
import pandas as pd
import os
import scipy.signal as ss

os.chdir('../data/input_B10/')  # 改变当前工作目录。注意路径的分隔符
file_chdir = os.getcwd()  # 获取当前工作目录
fileTxt_list = []  # 新建一个列表，用于存放各个文件
for root, dirs, files in os.walk(file_chdir):  # 读取一个文件夹里所有文件的固定用法，os.walk
    for file in files:
        if os.path.splitext(file)[1] == '.txt':  # 加一个判断是否为txt，相当于容错
            fileTxt_list.append(file)  # 将文件添加到列表中

k = 1

for txt in fileTxt_list:
    # 读取数据，只读取两列（因为用tab当作分隔符，会读出三列，但最后一列是没有的），用第一列时间作为索引,实际数据只有一列
    data = pd.read_csv(txt, sep='\t', index_col=u'Time - Plot 0', header=0, usecols=[0, 1])
    data = data.values  # 将dataFrame中 的数据转换成数组
    resultTemp_matrix = np.zeros([1001], float)  # 创建一个1000列的二维数组（矩阵）(1000是所有人为规定的特征集大小)

    '''这里的高度设置，要根据实际波形决定，尽量使这1000个数据，具有大范围的周期性'''
    max_index, max_value = ss.find_peaks(data[:, 0], height=0.0815, distance=200)  # 识别峰值，设置限制，高度为，距离为200  max_index为最高点下标，max_value为最高点的值
    time_index = max_index * 0.000005  # max_index是从1开始的下标,所以乘上一个时间间隔，得到实际时间下标

    temp = data[max_index[0]: max_index[0] + 1000].T  # 从第一个峰值开始，往后数1000个数据，将这些数据给result_matrix矩阵中
    resultTemp_matrix[:temp.size] = temp

    # 通过k值，判断是否为第一个，不是的话，将每个文件得到的数组进行连接，得到一个大数组，里面包括了所有的数据
    if k == 1:
        result_matrix = resultTemp_matrix
    else:
        result_matrix = np.vstack((result_matrix, resultTemp_matrix))  # np.vstack（）垂直拼接两个数组，这种垂直拼接与原来两个数组的维度无关，而concatenate()方法，拼接后的矩阵维度，和原维度相同，比如两个一维矩阵，拼接后仍是一维，而两个多维矩阵，拼接后仍是多维
    k = k + 1

result_matrix[:, 1000] = 8  # 写标签

# print(result_matrix)

np.savetxt('../output_B10/output.csv', result_matrix, delimiter=",")  # 将数组存储到output中
b = np.loadtxt('../output_B10/output.csv', delimiter=",")  # 读取数组（读得数据为数组形式）
print(b)


