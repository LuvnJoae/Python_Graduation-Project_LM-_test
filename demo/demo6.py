import matplotlib.pyplot as plt


for i in range(0, 10):
    datafile = '../data/input_B10/B10-10-000'+ str(i) +'.txt'
    data = pd.read_csv(datafile, sep='\t', index_col=u'Time - Plot 0', header=0,usecols=[0, 1])
    data = data.values  # 将dataFrame中 的数据转换成数组

    x = range(2001)
    y = data[:]

    plt.plot(x, y, 'b',label='time')
    plt.title('000' + str(i))
    plt.show()
