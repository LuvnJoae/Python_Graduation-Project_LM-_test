# 将 良品 与 不良品B1 数据相结合

import numpy as np
import pandas as pd
import os
import scipy.signal as ss

data_0 = np.loadtxt('../data/output/output.csv', delimiter=",") # 读取 良品数据 数组（读得数据为数组形式）
data_B1 = np.loadtxt('../data/output_false_B1All/output.csv', delimiter=",") # 读取 不良品B1数据 数组（读得数据为数组形式）

data = np.vstack((data_0, data_B1))

np.savetxt('../data/output_0&B1_All/output.csv', data, delimiter=",")# 将数组存储到output中
b = np.loadtxt('../data/output_0&B1_All/output.csv', delimiter=",") # 读取数组（读得数据为数组形式）
print(b)
