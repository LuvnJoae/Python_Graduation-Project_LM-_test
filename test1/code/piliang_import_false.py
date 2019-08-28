# 数据周期化处理，第一步
import numpy as np
import pandas as pd
import os
import scipy.signal as ss


os.chdir('D:\\PythonProject\\Test\\test1\\data\\input_false_1\\',)# 改变当前工作目录。注意路径的分隔符
file_chdir = os.getcwd() # 获取当前工作目录
fileTxt_list = [] # 新建一个列表，用于存放各个文件
for root, dirs, files in os.walk(file_chdir): # 读取一个文件夹里所有文件的固定用法，os.walk
    for file in files:
        if os.path.splitext(file)[1] == '.txt': # 加一个判断是否为txt，相当于容错
            fileTxt_list.append(file) # 将文件添加到列表中

k = 1
for txt in fileTxt_list:
    # 读取数据，只读取两列（因为用tab当作分隔符，会读出三列，但最后一列是没有的），用第一列时间作为索引,实际数据只有一列
    data = pd.read_csv(txt, sep='\t', index_col=u'Time - Plot 0', header=0, usecols=[0, 1])
    data = data.values  # 将dataFrame中 的数据转换成数组
    resultTemp_matrix = np.zeros([6, 320], float)  # 创建一个6行，304列的二维数组（矩阵）(304是所有数据统计下来的单一周期的最大值)

    max_index, max_value = ss.find_peaks(data[:, 0], height=0.1,distance=200)  # 设置每个周期限制，高度为0.06，距离为200  max_index为最高点下标，max_value为最高点的值
    time_index = max_index * 0.000005  # max_index是从1开始的下标,所以乘上一个时间间隔，得到实际时间下标

    for i in range(6): # 分别将6个周期的值存入到数组中。
        temp = data[max_index[i]: max_index[i + 1]].T
        resultTemp_matrix[i, : temp.size] = temp

    # 判断是否为第一个，不是的话，将每个文件得到的数组进行连接，得到一个大数组，里面包括了所有的数据
    if k == 1:
        result_matrix = resultTemp_matrix
    else:
        result_matrix = np.concatenate((result_matrix, resultTemp_matrix), axis=0)
    k = k + 1

# print(result_matrix)
# 以数组形式，保存到txt中
np.savetxt('D:\\PythonProject\\Test\\test1\\data\\output_false_1\\output.csv', result_matrix, delimiter=",")
b = np.loadtxt('D:\\PythonProject\\Test\\test1\\data\\output_false_1\\output.csv', delimiter=",")
print(b)
