import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.messagebox
from tkinter import filedialog
from tkinter import font
import pandas as pd
import psycopg2
from pandastable import TableModel
from pandastable import PlotViewer
from pandastable import Table
import re
from capexplain.explain.explanation import ExplanationGenerator
from capexplain.explain.explanation import ExplConfig
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from Plotting import Plotter
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s line %(lineno)d: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)



class Exp_Frame:

	def __init__(self,question_df=None, explanation_df=None, exp_chosen_row=None, none_drill_down_df=None, drill_down_df=None, data_convert_dict=None):

		self.win = Toplevel()
		self.win.geometry("%dx%d%+d%+d" % (1580, 900, 250, 125))
		self.win.wm_title("Explanation Detail")		
		self.win_frame = Frame(self.win)
		self.win_frame.pack(fill=BOTH,expand=True)

		self.question_df = question_df.astype(object)
		self.explanation_df = explanation_df.astype(object)
		self.exp_chosen_row = exp_chosen_row.astype(object)
		self.none_drill_down_df = none_drill_down_df.astype(object)
		self.drill_down_df = drill_down_df.astype(object)
		self.data_convert_dict=data_convert_dict
		self.drill_exist=False

		self.relevent_pattern = self.exp_chosen_row['From_Pattern']
		self.rel_pattern_part = self.relevent_pattern.split(':')[0].split('=')[0].strip('[')
		self.rel_pattern_pred = self.relevent_pattern.split(':')[1].split(' \u2933 ')[0]
		self.rel_pattern_agg = self.relevent_pattern.split(':')[1].split(' \u2933 ')[1]
		self.rel_pattern_model = self.exp_chosen_row['relevent_model']
		self.rel_param = self.exp_chosen_row['relevent_param']
		self.rel_pattern_part_list = self.rel_pattern_part.split(',')
		self.rel_pattern_pred_list = self.rel_pattern_pred.split(',')
		self.exp_tuple_score = float(self.exp_chosen_row['Score'])
		self.drill_attr = [self.exp_chosen_row['Drill_Down_To']]
		self.drill_model = self.exp_chosen_row['refinement_model']
		self.drill_param = self.exp_chosen_row['drill_param']


	# configure the frame structure according the exp type

		if(drill_down_df is None):

			self.win_frame.columnconfigure(0,weight=2)
			self.win_frame.columnconfigure(1,weight=3)
			self.win_frame.rowconfigure(0,weight=8)
			self.win_frame.rowconfigure(1,weight=1)

			self.Quit_Button = Button(self.win_frame, text="Quit",width=10, height=4, command=self.win.destroy)
			self.Quit_Button.grid(column=0,row=1)

			self.rel_graph_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.rel_graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')

			self.exp_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.exp_frame.grid(column=0,row=0,sticky='nesw')

			self.rel_figure = Figure(figsize=(5,5),dpi=130)

			self.rel_canvas = FigureCanvasTkAgg(self.rel_figure,self.rel_graph_frame)
			self.rel_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
			self.rel_toolbar = NavigationToolbar2Tk(self.rel_canvas,self.rel_graph_frame)
			self.rel_toolbar.update()
			self.rel_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
		
		else:

			self.drill_exist = True
			self.win_frame.columnconfigure(0,weight=1)
			self.win_frame.columnconfigure(1,weight=4)
			self.win_frame.columnconfigure(2,weight=5)
			self.win_frame.rowconfigure(0,weight=6)
			self.win_frame.rowconfigure(1,weight=2)
			self.win_frame.rowconfigure(2,weight=1)

			self.Quit_Button = Button(self.win_frame, text="Quit",width=10,height=4, command=self.win.destroy)
			self.Quit_Button.grid(column=0,row=2)

			self.rel_graph_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.rel_graph_frame.grid(column=0,columnspan=2,row=0,sticky='nesw')

			self.drill_graph_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.drill_graph_frame.grid(column=2,row=0,sticky='nesw')

			self.rel_figure = Figure(figsize=(5,5),dpi=130)
			self.rel_canvas = FigureCanvasTkAgg(self.rel_figure,self.rel_graph_frame)
			self.rel_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
			self.rel_toolbar = NavigationToolbar2Tk(self.rel_canvas,self.rel_graph_frame)
			self.rel_toolbar.update()
			self.rel_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

			self.drill_figure = Figure(figsize=(5,5),dpi=130)
			self.drill_canvas = FigureCanvasTkAgg(self.drill_figure,self.drill_graph_frame)
			self.drill_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
			self.drill_toolbar = NavigationToolbar2Tk(self.drill_canvas,self.drill_graph_frame)
			self.drill_toolbar.update()
			self.drill_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

			self.exp_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.exp_frame.grid(column=1,columnspan=2,row=1,rowspan=2,sticky='nesw')


	def load_exp_graph(self):

		if(self.drill_exist==False):
			print('as expected!')
			self.load_rel_exp_graph()
		else:
			self.load_rel_question_graph()
			self.load_drill_exp_graph()


	def load_rel_exp_graph(self):

		# self.rel_pattern_model, self.rel_param, self.question_df, self.explanation_df, 

		if(self.rel_pattern_model=='const'):
			if(len(self.rel_pattern_pred_list)==1):
				self.rel_plotter = Plotter(figure=self.rel_figure,data_convert_dict=self.data_convert_dict,mode='2D')
				const=round(float(self.rel_param),2)

				x=self.rel_pattern_pred_list[0]
				y=self.rel_pattern_agg

				self.rel_plotter.plot_2D_const(const)
				# self.rel_plotter.plot_categorical_scatter_2D(x,y)
				none_drill_down_exp_df = self.none_drill_down_df[[x,y]]
				logger.debug(none_drill_down_exp_df)

				question_df = self.question_df[[x,y]]
				logger.debug(question_df)

				explanation_df = self.explanation_df[[x,y]]
				logger.debug(explanation_df)

				self.rel_plotter.plot_2D_scatter(question_df,x=x,y=y,color='r',marker='P',size=200,zorder=10)
				self.rel_plotter.plot_2D_scatter(explanation_df,x=x,y=y,color='g',marker='D',size=200,zorder=5)
				self.rel_plotter.plot_2D_scatter(none_drill_down_exp_df,x=x,y=y,zorder=0)
				self.rel_plotter.set_x_label(x)
				self.rel_plotter.set_y_label(y)
				self.rel_plotter.set_title("pattern graph")
		
			else:

				self.rel_plotter = Plotter(figure=self.rel_figure,data_convert_dict=self.data_convert_dict,mode='3D')

				x = self.rel_pattern_pred_list[0]
				y = self.rel_pattern_pred_list[1]
				z = self.rel_pattern_agg

				const = round(self.rel_param,2)

				none_drill_down_exp_df = self.none_drill_down_df[[x,y,z]]
				logger.debug(none_drill_down_exp_df)

				question_df = self.question_df[[x,y,z]]
				logger.debug(question_df)

				explanation_df = self.explanation_df[[x,y,z]]
				logger.debug(explanation_df)

				pattern_only_df = pd.concat([none_drill_down_exp_df,question_df,explanation_df]).drop_duplicates(keep=False)

				self.rel_plotter.plot_3D_const(none_drill_down_exp_df,x=x,y=y,z_value=const)
				self.rel_plotter.plot_3D_scatter(none_drill_down_exp_df,x=x,y=y,z=z,alpha=0)
				self.rel_plotter.plot_3D_scatter(pattern_only_df,x=x,y=y,z=z)
				self.rel_plotter.plot_3D_scatter(question_df,x=x,y=y,z=z,color='b',marker='s',size=200)
				self.rel_plotter.plot_3D_scatter(explanation_df,x=x,y=y,z=z,color='r',marker='P',size=200)

				self.rel_plotter.set_x_label(x)
				self.rel_plotter.set_y_label(y)
				self.rel_plotter.set_z_label(z)
				self.rel_plotter.set_title("pattern graph")

		elif(self.rel_pattern_model=='linear'):
			if(len(self.rel_pattern_pred_list)==1):

				self.rel_plotter = Plotter(figure=self.rel_figure,data_convert_dict=self.data_convert_dict,mode='2D')
				x = self.rel_pattern_pred_list[0]
				y = self.rel_pattern_agg

				intercept_value = self.rel_param['Intercept']
				slope_name = list(self.rel_param)[1]
				slope_value = float(self.rel_param[slope_name])

				draw_line_df = self.none_drill_down_df[[x]]

				none_drill_down_exp_df = self.none_drill_down_df[[x,y]]
				logger.debug(none_drill_down_exp_df)

				question_df = self.question_df[[x,y]]
				logger.debug(question_df)

				explanation_df = self.explanation_df[[x,y]]
				logger.debug(explanation_df)

				self.rel_plotter.plot_2D_linear(draw_line_df,slope=slope_value,intercept=intercept_value)
				self.rel_plotter.plot_2D_scatter(none_drill_down_df,x=x,y=y)
				self.rel_plotter.plot_2D_scatter(question_df,x=x,y=y,zorder=1)
				self.rel_plotter.plot_2D_scatter(explanation_df,x=x,y=y,zorder=2)
				self.rel_plotter.set_x_label(x)
				self.rel_plotter.set_y_label(y)
				self.rel_plotter.set_title("pattern graph")

		self.rel_canvas.draw()


	def load_rel_question_graph(self):

		if(self.rel_pattern_model=='const'):
			if(len(self.rel_pattern_pred_list)==1):
				self.rel_plotter = Plotter(figure=self.rel_figure,data_convert_dict=self.data_convert_dict,mode='2D')
				const=round(float(self.rel_param),2)

				x=self.rel_pattern_pred_list[0]
				y=self.rel_pattern_agg

				self.rel_plotter.plot_2D_const(const)
				# self.rel_plotter.plot_categorical_scatter_2D(x,y)

				common_cols = self.rel_pattern_part_list + self.rel_pattern_pred_list

				logger.debug("common_cols for question is ")
				print(common_cols)

				question_df = pd.merge(self.none_drill_down_df,self.question_df,on=common_cols)

				logger.debug("question_df is ")
				print(question_df)


				question_df = question_df.rename(index=str, columns={(y+"_x"): y,(x+"_x"):x})
				question_df = question_df[[x,y]]

				logger.debug(question_df)

				self.rel_plotter.plot_2D_scatter(question_df,x=x,y=y,color='r',marker='P',size=200,zorder=10)
				self.rel_plotter.plot_2D_scatter(self.none_drill_down_df,x=x,y=y,zorder=0)
				self.rel_plotter.set_x_label(x)
				self.rel_plotter.set_y_label(y)
				self.rel_plotter.set_title("pattern graph")
		
			else:


				pass

				# self.rel_plotter = Plotter(figure=self.rel_figure,data_convert_dict=self.data_convert_dict,mode='3D')

				# x = self.rel_pattern_pred_list[0]
				# y = self.rel_pattern_pred_list[1]
				# z = self.rel_pattern_agg

				# const = round(float(self.rel_param),2)

				# none_drill_down_exp_df = self.none_drill_down_df[[x,y,z]]
				# logger.debug(none_drill_down_exp_df)

				# question_df = self.question_df[[x,y,z]]
				# logger.debug(question_df)


				# pattern_only_df = pd.concat([none_drill_down_exp_df,question_df]).drop_duplicates(keep=False)

				# self.rel_plotter.plot_3D_const(none_drill_down_exp_df,x=x,y=y,z_value=const)
				# self.rel_plotter.plot_3D_scatter(none_drill_down_exp_df,x=x,y=y,z=z,alpha=0)
				# self.rel_plotter.plot_3D_scatter(pattern_only_df,x=x,y=y,z=z)
				# self.rel_plotter.plot_3D_scatter(question_df,x=x,y=y,z=z,color='b',marker='s',size=200)

				# self.rel_plotter.set_x_label(x)
				# self.rel_plotter.set_y_label(y)
				# self.rel_plotter.set_z_label(z)
				self.rel_plotter.set_title("pattern graph")

		elif(self.rel_pattern_model=='linear'):
			if(len(self.rel_pattern_pred_list)==1):

				self.rel_plotter = Plotter(figure=self.rel_figure,data_convert_dict=self.data_convert_dict,mode='2D')
				x = self.rel_pattern_pred_list[0]
				y = self.rel_pattern_agg

				intercept_value = self.rel_param['Intercept']
				slope_name = list(self.rel_param)[1]
				slope_value = float(self.rel_param[slope_name])

				draw_line_df = self.none_drill_down_df[[x]]

				common_cols = self.rel_pattern_part_list+ self.rel_pattern_pred_list

				question_df = pd.merge(self.none_drill_down_df,self.question_df,on=common_cols)

				question_df = question_df.rename(index=str, columns={(y+"_x"): y,(x+"_x"):x})
				question_df = question_df[[x,y]]

				self.rel_plotter.plot_2D_linear(draw_line_df,slope=slope_value,intercept=intercept_value)
				self.rel_plotter.plot_2D_scatter(self.none_drill_down_df,x=x,y=y)
				self.rel_plotter.plot_2D_scatter(question_df,x=x,y=y,color='r',marker='P',size=200,zorder=1)
				self.rel_plotter.set_x_label(x)
				self.rel_plotter.set_y_label(y)
				self.rel_plotter.set_title("pattern graph")

		self.rel_canvas.draw()

	def load_drill_exp_graph(self):

		if(self.drill_model=='const'):

			if(len(self.drill_attr)==1):
				self.drill_plotter = Plotter(figure=self.drill_figure,data_convert_dict=self.data_convert_dict,mode='2D')
				const=round(float(self.drill_param),2)

				x=self.rel_pattern_pred_list[0]
				y=self.rel_pattern_agg

				self.drill_plotter.plot_2D_const(const)
				# self.drill_plotter.plot_categorical_scatter_2D(x,y)

				common_cols = self.rel_pattern_part_list+ self.drill_attr+self.rel_pattern_pred_list

				logger.debug('common_cols:')
				print(common_cols)

				logger.debug("self.explanation_df")
				print(self.explanation_df)

				logger.debug("self.drill_down_df")
				print(self.drill_down_df)

				logger.debug("self.drill_down_df data types:")
				print(self.drill_down_df.dtypes)

				logger.debug("self.explanation_df data types:")
				print(self.explanation_df.dtypes)


				self.drill_plotter.plot_2D_scatter(self.explanation_df,x=x,y=y,color='r',marker='P',size=200,zorder=10)
				self.drill_plotter.plot_2D_scatter(self.drill_down_df,x=x,y=y,zorder=0)
				self.drill_plotter.set_x_label(x)
				self.drill_plotter.set_y_label(y)
				self.drill_plotter.set_title("pattern graph")
		
			else:
				pass

				# self.drill_plotter = Plotter(figure=self.drill_figure,data_convert_dict=self.data_convert_dict,mode='3D')

				# x = self.rel_pattern_pred_list[0]
				# y = self.rel_pattern_pred_list[1]
				# z = self.rel_pattern_agg

				# const = round(float(self.drill_param),2)

				# drill_down_df = self.drill_down_df[[x,y,z]]
				# logger.debug(drill_down_df)

				# explanation_df = self.explanation_df[[x,y,z]]
				# logger.debug(explanation_df)


				# pattern_only_df = pd.concat([drill_down_df,explanation_df]).drop_duplicates(keep=False)

				# self.drill_plotter.plot_3D_const(drill_down_df,x=x,y=y,z_value=const)
				# self.drill_plotter.plot_3D_scatter(drill_down_df,x=x,y=y,z=z,alpha=0)
				# self.drill_plotter.plot_3D_scatter(pattern_only_df,x=x,y=y,z=z)
				# self.drill_plotter.plot_3D_scatter(explanation_df,x=x,y=y,z=z,color='b',marker='s',size=200,label='Explanation')

				# self.drill_plotter.set_x_label(x)
				# self.drill_plotter.set_y_label(y)
				# self.drill_plotter.set_z_label(z)
				# self.drill_plotter.set_title("pattern graph")

				# self.drill_canvas.draw()


		elif(self.drill_model=='linear'):

			if(len(self.drill_attr)==1):
				self.drill_plotter = Plotter(figure=self.drill_figure,data_convert_dict=self.data_convert_dict,mode='2D')
				x = self.drill_attr[0]
				y = self.rel_pattern_agg

				intercept_value = self.drill_param['Intercept']
				slope_name = list(self.drill_param)[1]
				slope_value = float(self.drill_param[slope_name])

				draw_line_df = self.none_drill_down_df[[x]]

				common_cols = self.rel_pattern_part_list+self.drill_attr

				explanation_df = pd.merge(self.drill_down_df,self.explanation_df,on=common_cols)

				explanation_df = explanation_df.rename(index=str, columns={(y+"_x"): y,(x+"_x"):x})

				explanation_df = explanation_df[[x,y]]

				logger.debug(explanation_df)
				self.drill_plotter.plot_2D_linear(draw_line_df,slope=slope_value,intercept=intercept_value)
				self.drill_plotter.plot_2D_scatter(self.drill_down_df,x=x,y=y)
				self.drill_plotter.plot_2D_scatter(explanation_df,x=x,y=y,zorder=1)
				self.drill_plotter.set_x_label(x)
				self.drill_plotter.set_y_label(y)
				self.drill_plotter.set_title("pattern graph")

		self.drill_canvas.draw()


def main():
	
	relevent_pattern_df = pd.DataFrame({'name':['Boris Glavic','Boris Glavic','Boris Glavic','Boris Glavic'],'venue':['sigmod','vldb','kdd','icde'],'year':['2013','2014','2016','2018'],'pubcount':[5,4,3,10]})

	drill_down_df = pd.DataFrame({'name':['Boris Glavic','Boris Glavic','Boris Glavic','Boris Glavic'],'year':['2013','2014','2016','2018'],'venue':['sigmod','vldb','kdd','icde'],'pubcount':[5,4,1,7]})
	question_df = pd.DataFrame({'name':['Boris Glavic'],'venue':['icde'],'year':['2018'],'pubcount':[10]})
	explanation_df = pd.DataFrame({'name':['Boris Glavic'],'venue':['kdd'],'year':['2016'],'pubcount':[1]})

	exp_chosen_row = pd.DataFrame({'Explanation_Tuple':['Boris Glavic,kdd,2016,3'],'Score':[7.34],
		'From_Pattern':['[name=Boris Glavic]:year \u2933 pubcount'],'Drill_Down_To':['venue'],'Distance':[11.36],
		'Outlierness':[0],'Denominator':[-0.83],'relevent_model':['const'],'relevent_param':[4.5],'refinement_model':['const'],
		'drill_param':[3.2]})

	data_type_dict = {'name': 'str', 'venue': 'str', 'year': 'numeric', 'pubcount': 'numeric'}


	root = Tk()

	exp_frame_1 = Exp_Frame(question_df=question_df, explanation_df=explanation_df, exp_chosen_row=exp_chosen_row, none_drill_down_df=relevent_pattern_df,
		drill_down_df=drill_down_df, data_convert_dict=data_type_dict)

	exp_frame_1.load_exp_graph()

	root.mainloop()
	
if __name__ == '__main__':
	main()

