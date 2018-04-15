import numpy as np

a = [[1,2,3,4]]
b = np.reshape(a,(2,-1))
b = np.array(b)
print(b.shape)
