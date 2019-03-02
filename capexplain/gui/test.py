import re
import pandas as pd

# class test:

# 	def __init__(self,size):
# 		self.size = size

# 	def __getitem__(self, key):
# 	    if key not in self.__dict__:
# 	        raise AttributeError("No such attribute: " + key)
# 	    return self.__dict__[key]

# 	# overwrite __setitem__ to allow dictory style setting of options
# 	def __setitem__(self,key,value):
# 	    if key not in self.__dict__:
# 	        raise AttributeError("No such attribute: " + key)
# 	    self.__dict__[key] = value

# 	def print_dict(self):
# 		print(self.__dict__)


# if __name__ == '__main__':

# 	test1 = test(4)

# 	test1.print_dict();

# 	print(test1['size'])

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt
import random

def fun(x, y):
  return x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
print("X")
print(X)
print("Y")
print(Y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
print("zs")
print(zs)

Z = zs.reshape(X.shape)
print("Z")
print(Z)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()