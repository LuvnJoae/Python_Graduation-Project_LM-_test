import numpy as np

a = np.array([1,2])
print(a)
b = np.array([3,4])
print(b)

c = np.concatenate((a, b), axis=0)
print(a)