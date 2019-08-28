#-*- coding: utf-8 -*-
#拉格朗日插值代码
import pandas as pd #导入数据分析库Pandas
import numpy as np
from scipy.interpolate import lagrange #导入拉格朗日插值函数

inputfile = 'D:\\PythonProject\\Test\\test1\\data\\output\\output.csv' #输入数据路径,需要使用Excel格式；
outputfile = 'D:\\PythonProject\\Test\\test1\\data\\output\\output_processed.csv' #输出数据路径,需要使用Excel格式

# 输入
data = pd.read_csv(inputfile,header=None,sep=',',usecols=[294])

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=3):
  y = s.reindex(list(range(n-k, n)) + list(range(n+1, n+1+k))) #取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) #插值并返回插值结果

#逐个元素判断是否需要插值
for i in data.columns:
  for j in range(len(data)):
    if (data[i]==0)[j]: #如果为空即插值。
      data[i][j] = ployinterp_column(data[i], j)


data.to_csv(outputfile, header=None, index=None) #输出结果
#输出结果