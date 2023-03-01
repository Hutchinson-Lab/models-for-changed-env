

# print([0]*20)


import numpy as np

a = np.arange(0,10,1).reshape(5,2)

idx = [1,2]

print(a)

a[idx,1] = np.where(a[idx,1]<5, a[idx,1], 0)

print(a)