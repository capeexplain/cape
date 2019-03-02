"""
define the plotting functions used in "CAPE"
"""
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 

class Plotter:

	def __init__(self,figure):

		self.figure = figure
		self.a = self.figure.add_subplot(111)

	def set_x_label(self,label_name):

		self.a.set_xlabel(label_name)


	def set_y_label(self,label_name):

		self.a.set_ylabel(label_name)


	def set_z_label(self,label_name):

		self.a.set_zlabel(label_name)


	def set_title(self,title_name):

		self.a.set_title(title_name)


	def plot_const(self,const_value):

		self.a.axhline(const_value,c="red",linewidth=2,label='constant = '+str(const_value))
		self.a.legend(loc='best')


	def plot_const_3D(self,x,y,z):

		self.a = self.figure.gca(projection='3d')
		Xuniques, X = np.unique(x, return_inverse=True)
		Yuniques, Y = np.unique(y, return_inverse=True)
		x_range=np.arange(X.min(),X.max()+1)
		y_range=np.arange(Y.min(),Y.max()+1)
		X1, Y1 = np.meshgrid(x_range, y_range)
		zs = np.array([z for x1,y1 in zip(np.ravel(X1), np.ravel(Y1))])
		Z = zs.reshape(X1.shape)
		self.a.plot_surface(X1, Y1, Z,color='r',linewidth=2)
		self.a.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)
		self.a.set(yticks=range(len(Yuniques)), yticklabels=Yuniques)


	def plot_linear_2D(self,x,slope,intercept):

		self.a.legend(loc='best')
		var_min = pd.to_numeric(x.min()).item()
		var_max = pd.to_numeric(x.max()).item()
		X1 = np.linspace(var_min-3,var_max+3,100)
		X = x.astype('float64')
		y_vals = slope * X1 + intercept
		self.a.plot(X1, y_vals, c='r',linewidth=2,label='Model')


	def plot_categorical_scatter_2D(self,x,y):

		Xuniques, X = np.unique(x, return_inverse=True)
		Y = y
		self.a.scatter(X, Y, s=60, c='b')
		self.a.legend(loc='best')
		self.a.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)


	def plot_categorical_scatter_3D(self,x,y,z):

		self.a = self.figure.gca(projection='3d')
		Xuniques, X = np.unique(x, return_inverse=True)
		Yuniques, Y = np.unique(y, return_inverse=True)
		self.a.scatter(X, Y, z,s=60, c='b')

	def plot_numerical_scatter_2D(self,x,y):

		self.a.scatter(x, y, s=60, c='b')
		self.a.legend(loc='best')












		


