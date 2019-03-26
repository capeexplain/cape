"""
define the plotting functions used in "CAPE"
"""
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from math import floor,ceil
import re


coded = re.compile('(coded\_).*')

class Plotter:

	def __init__(self,figure,data_convert_dict,mode='2D'):
		
		self.data_convert_dict = data_convert_dict
		self.figure = figure
		self.legend_proxies_shape= []  # for 3d legend specifically
		self.legend_proxies_name = [] # for 3d legend specifically

		if(mode=='2D'):
			self.a = self.figure.add_subplot(111)
		elif(mode=='3D'):
			self.a= self.figure.gca(projection='3d')


		self.cur_x_range = [0]
		self.cur_y_range = [0]
		self.cur_z_range = [0]

		self.encoding_list = []


	def df_type_conversion(self,data_frame):  # convert column data types based on database information(plot_data_convert_dict)
		numeric_cols = []
		str_cols = []

		for n in list(data_frame):
			if coded.search(n) is not None:
				continue
			elif(self.data_convert_dict[n]=='numeric'):
				numeric_cols.append(n) 
			elif(self.data_convert_dict[n]=='str'):
				str_cols.append(n)

		if(len(numeric_cols)>0):
			data_frame[numeric_cols] = data_frame[numeric_cols].apply(lambda x : pd.to_numeric(x))

		if(len(str_cols)>0):
			for n in str_cols:
				if('coded_'+n in data_frame.columns):
					continue

				if(len(self.encoding_list)>0):
					for x in self.encoding_list:
						if(('coded_'+n) == x['dict_name']):
							data_frame['coded_'+n] = data_frame[n].map(x)

				if(('coded_'+n) not in data_frame.columns):
					data_frame['coded_'+n] = data_frame[n].astype('category').cat.codes
					dict1 = dict(zip(data_frame[n],data_frame['coded_'+n]))
					dict1['dict_name'] = 'coded_'+n
					self.encoding_list.append(dict1)

		print("encoding_list is :")
		print(self.encoding_list)

		return data_frame


	def set_x_label(self,label_name):
		self.a.set_xlabel(label_name)


	def set_y_label(self,label_name):
		self.a.set_ylabel(label_name)


	def set_z_label(self,label_name):
		self.a.set_zlabel(label_name)


	def set_title(self,title_name):
		self.a.set_title(title_name)


	def plot_2D_const(self,const_value,label=None):

		self.a.axhline(const_value,c="red",linewidth=2,label=label)
		self.a.legend(loc='best')


	def plot_3D_const(self,df,x=None,y=None,z_value=None,label=None): # df is the dataframe containing the columns to be drawn

		df = self.df_type_conversion(df)

		if ('coded_'+x) in df.columns:
			x_range=np.arange(df['coded_'+x].min(),df['coded_'+x].max()+1,1)
		else:
			x_range=np.arange(df[x].min(),df[x].max()+1,1)

		if ('coded_'+y) in df.columns:
			y_range=np.arange(df['coded_'+y].min(),df['coded_'+y].max()+1,1)
		else:
			y_range=np.arange(df[y].min(),df[y].max()+1,1)

		X1, Y1 = np.meshgrid(x_range, y_range)
		zs = np.array([z_value for x1,y1 in zip(np.ravel(X1), np.ravel(Y1))])
		Z = zs.reshape(X1.shape)
		self.a.plot_surface(X1, Y1, Z,color='r',linewidth=2,label=label)

		if ('coded_'+x) in df.columns:
			self.a.set(xticks=df['coded_'+x].values, xticklabels=df[x])

		else:
			self.a.set(xticks=list(range(floor(df[x].values.min()),ceil(df[x].values.max()+1,3))))

		if ('coded_'+y) in df.columns:
			self.a.set(yticks=df['coded_'+y].values, yticklabels=df[y])
		else:
			self.a.set(yticks=list(range(floor(df[y].values.min()),ceil(df[y].values.max()+1),5)))

		plane_proxy_shape = plt.Rectangle((0, 0), 1, 1, fc="r")
		plane_proxy_name = label
		self.legend_proxies_shape.append(plane_proxy_shape)
		self.legend_proxies_name.append(plane_proxy_name)
		self.a.legend(self.legend_proxies_shape,self.legend_proxies_name)


	def plot_2D_linear(self,x,slope,intercept,label=None):  # x:df column; slope,intercept:numeric values
		var_min = pd.to_numeric(x.min()).item()
		var_max = pd.to_numeric(x.max()).item()
		X1 = np.linspace(var_min-1,var_max+1,100)
		X = x.astype('float64')
		y_vals = slope * X1 + intercept
		self.a.plot(X1, y_vals, c='r',linewidth=2,label=label)
		self.a.legend(loc='best')

	def plot_3D_linear(self,x,y,x_weight,y_weight):
		pass 


	def plot_2D_scatter(self,df,x=None,y=None,color='g',marker='o',size=60,zorder=0,alpha=1,label=None): # x,y are 2 df column names

		df = self.df_type_conversion(df)

		if ('coded_'+x) in df.columns:
			X=df['coded_'+x]
		else:
			X=df[x]

		if ('coded_'+y) in df.columns:
			Y=df['coded_'+y]
		else:
			Y=df[y]

		self.a.scatter(X, Y, s=size, c=color,marker=marker,zorder=zorder,alpha=1,label=label)
		self.a.legend(loc='best')


		if ('coded_'+x) in df.columns:

			x_ticks = df['coded_'+x].values
			if(len(x_ticks)>len(self.cur_x_range)):
				self.a.set(xticks=x_ticks, xticklabels=df[x])
				self.cur_x_range = x_ticks


		else:
			x_ticks = list(range(floor(df[x].values.min()),ceil(df[x].values.max()+1),3))
			if(len(x_ticks)>len(self.cur_x_range)):
				self.a.set(xticks=x_ticks)
				self.cur_x_range = x_ticks


		if ('coded_'+y) in df.columns:			
			y_ticks = df['coded_'+y].values
			if(len(y_ticks)>len(self.cur_y_range)):
				self.a.set(yticks=y_ticks, yticklabels=df[y])
				self.cur_y_range = y_ticks

		else:
			y_ticks = list(range(floor(df[y].values.min()),ceil(df[y].values.max()+1),5))
			if(len(y_ticks)>len(self.cur_y_range)):
				self.a.set(yticks=y_ticks)
				self.cur_y_range = y_ticks


	def plot_3D_scatter(self,df,x,y,z,color='g',marker='o',size=60,zorder=0,alpha=1,label=None): # x,y,z are 3 df columns

		df = self.df_type_conversion(df)

		if ('coded_'+x) in df.columns:
			X=df['coded_'+x]
		else:
			X=df[x]

		if ('coded_'+y) in df.columns:
			Y=df['coded_'+y]
		else:
			Y=df[y]

		if ('coded_'+z) in df.columns:
			Z=df['coded_'+z]
		else:
			Z=df[z]

		self.a.scatter(X, Y, Z, s=size, c=color,marker=marker,zorder=zorder,alpha=alpha,label=label)
		
		if ('coded_'+x) in df.columns:
			x_ticks = df['coded_'+x].values
			if(len(x_ticks)>len(self.cur_x_range)):
				self.a.set(xticks=x_ticks, xticklabels=df[x])
				self.cur_x_range = x_ticks


		else:
			x_ticks = list(range(floor(df[x].values.min()),ceil(df[x].values.max()+1),3))
			if(len(x_ticks)>len(self.cur_x_range)):
				self.a.set(xticks=x_ticks)
				self.cur_x_range = x_ticks

		if ('coded_'+y) in df.columns:
			
			y_ticks = df['coded_'+y].values
			if(len(y_ticks)>len(self.cur_y_range)):
				self.a.set(yticks=y_ticks, yticklabels=df[y])
				self.cur_y_range = y_ticks

		else:
			y_ticks = list(range(floor(df[y].values.min()),ceil(df[y].values.max()+1),5))
			if(len(y_ticks)>len(self.cur_y_range)):
				self.a.set(yticks=y_ticks)
				self.cur_y_range = y_ticks


		if ('coded_'+z) in df.columns:
			
			z_ticks = df['coded_'+z].values
			if(len(z_ticks)>len(self.cur_z_range)):
				self.a.set(zticks=z_ticks, zticklabels=df[z])
				self.cur_z_range = z_ticks

		else:
			z_ticks = list(range(floor(df[z].values.min()),ceil(df[z].values.max()+1)))
			if(len(z_ticks)>len(self.cur_z_range)):
				self.a.set(zticks=z_ticks)
				self.cur_z_range = z_ticks

		scatter_proxy_shape = plt.Rectangle((0, 0), 1, 1, fc=color)
		scatter_proxy_name = label
		self.legend_proxies_shape.append(scatter_proxy_shape)
		self.legend_proxies_name.append(scatter_proxy_name)
		self.a.legend(self.legend_proxies_shape,self.legend_proxies_name)


	def add_text(self,text_content=None):

		self.a.text(0.5, 0.5, text_content, horizontalalignment='center',
			verticalalignment='center', transform=self.a.transAxes)



def main():
	test_df = pd.DataFrame({'name':['sigmod','sigmod','bbb','ccc'],'year':['2013','2014','2016','2018'],'pubcount':[5,4,3,10]})

	data_type_dict = {'name': 'str', 'venue': 'str', 'year': 'numeric', 'pubcount': 'numeric'}

	plotter = Plotter(data_convert_dict=data_type_dict,mode='3D')
	threeD_df = test_df[['name','year','pubcount']]
	plotter.plot_3D_scatter(threeD_df,x='name',y='year',z='pubcount')
	plt.show()


	plotter_2 = Plotter(data_convert_dict=data_type_dict)
	twoD_df = test_df.loc[test_df['name']!='bbb'][['name','pubcount']]
	question_df = test_df.loc[test_df['name']=='bbb']
	twoD_question_df = question_df[['name','pubcount']]
	plotter_2.plot_2D_scatter(twoD_question_df,x='name',y='pubcount',color='r',marker='P',size=100)
	plotter_2.plot_2D_scatter(twoD_df,x='name',y='pubcount')
	plt.show()


	plotter_3 = Plotter(data_convert_dict=data_type_dict,mode='2D')
	twoD_df_1 = test_df[['year','pubcount']]
	question_df_1 = test_df.loc[test_df['name']=='bbb']
	explanation_df_1 = test_df.loc[test_df['name']=='ccc']
	twoD_question_df_1 = question_df[['year','pubcount']]
	twoD_explanation_df_1 = explanation_df_1[['year','pubcount']]
	plotter_3.plot_2D_scatter(twoD_question_df_1,x='year',y='pubcount',color='r',marker='P',size=250,zorder=1)
	plotter_3.plot_2D_scatter(twoD_explanation_df_1,x='year',y='pubcount',color='g',marker='*',size=250,zorder=2)
	plotter_3.plot_2D_scatter(twoD_df_1,x='year',y='pubcount',zorder=0)
	plt.show()


if __name__=='__main__':
	main()











