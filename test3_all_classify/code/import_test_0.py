''' 注意
        # 这个程序是用来生成 测试 文件的，可重复使用 ，用于生成有周期的数据文件
        # 更换测试案例时，要修改的地方有：
            os.chdir('../data/test/B1/') ： 更换目录
            max_index, max_value = ss.find_peaks(data[:, 0], height=0.08, distance=200) ： 更换峰值
            result_matrix[:, 1000] = 1  ： 更换标签（也可以不加）

        # 以含周期的方式 生成。
'''
import numpy as np
import pandas as pd
import os
import scipy.signal as ss

os.chdir('../data/test/B10')  # 改变当前工作目录。注意路径的分隔符
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

    max_index, max_value = ss.find_peaks(data[:, 0], height=0.0815, distance=200)  # 识别峰值，设置限制，高度为0.06，距离为200  max_index为最高点下标，max_value为最高点的值
    time_index = max_index * 0.000005  # max_index是从1开始的下标,所以乘上一个时间间隔，得到实际时间下标

    temp = data[max_index[0]: max_index[0] + 1000].T  # 从第一个峰值开始，往后数1000个数据，将这些数据给result_matrix矩阵中
    resultTemp_matrix[:temp.size] = temp

    # 通过k值，判断是否为第一个，不是的话，将每个文件得到的数组进行连接，得到一个大数组，里面包括了所有的数据
    if k == 1:
        result_matrix = resultTemp_matrix
    else:
        result_matrix = np.vstack((result_matrix, resultTemp_matrix))  # np.vstack（）垂直拼接两个数组，这种垂直拼接与原来两个数组的维度无关，而concatenate()方法，拼接后的矩阵维度，和原维度相同，比如两个一维矩阵，拼接后仍是一维，而两个多维矩阵，拼接后仍是多维
    k = k + 1

# result_matrix[:,:1000] = result_matrix[:,:1000] + 0.05
result_matrix[:, 1000] = 8  # 写标签

# print(result_matrix)

np.savetxt('output.csv', result_matrix, delimiter=",")  # 将数组存储到output中
b = np.loadtxt('output.csv', delimiter=",")  # 读取数组（读得数据为数组形式）
print(b)


