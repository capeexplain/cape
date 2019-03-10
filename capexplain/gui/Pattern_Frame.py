import pandas as pd
import psycopg2
import tkinter as tk
from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
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


class Local_Pattern_Frame:

	def __init__(self,chosen_row,pattern_data_df,agg_alias):

		self.chosen_row = chosen_row
		self.pattern_data_df = pattern_data_df
		self.agg_alias = agg_alias

		self.pop_up_frame = Toplevel()
		self.pop_up_frame.geometry("%dx%d%+d%+d" % (1300, 800, 250, 125))
		self.pop_up_frame.wm_title("Pattern Detail")

		self.win_frame = Frame(self.pop_up_frame)
		self.win_frame.pack(fill=BOTH,expand=True)
		self.win_frame.columnconfigure(0,weight=1)
		self.win_frame.columnconfigure(1,weight=1)
		self.win_frame.rowconfigure(0,weight=5)
		self.win_frame.rowconfigure(1,weight=1)

		b = ttk.Button(self.win_frame, text="Quit",width=10,command=self.pop_up_frame.destroy)
		b.grid(column=0,row=1)


	def load_pattern_graph(self,data_convert_dict=None):

		self.data_convert_dict = data_convert_dict
		graph_frame = Frame(self.win_frame)
		graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')
		self.figure = Figure(figsize=(5,5),dpi=130)
		canvas = FigureCanvasTkAgg(self.figure,graph_frame)
		canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
		toolbar = NavigationToolbar2Tk(canvas,graph_frame)
		toolbar.update()
		canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

		if(self.chosen_row['model']=='const'):
			if(len(self.chosen_row['variable'])==1):
				self.plotter = Plotter(figure=self.figure,data_convert_dict=self.data_convert_dict,mode='2D')
				variable_name= self.chosen_row['variable'][0]
				const=round(self.chosen_row['stats'],2)
				# x = self.pattern_data_df[variable_name]
				# y = self.pattern_data_df[self.agg_alias]
				self.plotter.plot_2D_const(const)
				# self.plotter.plot_categorical_scatter_2D(x,y)
				draw_df = self.pattern_data_df[[variable_name,self.agg_alias]]
				logger.debug(draw_df)

				self.plotter.plot_2D_scatter(draw_df,x=variable_name,y=self.agg_alias)
				self.plotter.set_x_label(variable_name)
				self.plotter.set_y_label(self.agg_alias)
				self.plotter.set_title("pattern graph")
		
			else:

				self.plotter = Plotter(figure=self.figure,data_convert_dict=self.data_convert_dict,mode='3D')
				x_name = self.chosen_row['variable'][0]
				y_name = self.chosen_row['variable'][1]
				const = self.chosen_row['stats']
				draw_const_df = self.pattern_data_df[[x_name,y_name]]
				draw_scatter_df = self.pattern_data_df[[x_name,y_name,self.agg_alias]]
				self.plotter.plot_3D_const(draw_const_df,x=x_name,y=y_name,z_value=const)
				self.plotter.plot_3D_scatter(draw_scatter_df,x=x_name,y=y_name,z=self.agg_alias)
				self.plotter.set_x_label(x_name)
				self.plotter.set_y_label(y_name)
				self.plotter.set_z_label(self.agg_alias)
				self.plotter.set_title("pattern graph")

		elif(self.chosen_row['model']=='linear'):
			if(len(self.chosen_row['variable'])==1):

				self.plotter = Plotter(figure=self.figure,data_convert_dict=self.data_convert_dict,mode='2D')
				variable_name = self.chosen_row['variable'][0]
				intercept_value = round((self.chosen_row['param']['Intercept']),2)
				slope_name = list(self.chosen_row['param'])[1]
				slope_value = float(self.chosen_row['param'][slope_name])
				# x = self.pattern_data_df[variable_name]
				# y = self.pattern_data_df[self.agg_alias]

				draw_line_df = self.pattern_data_df[[variable_name]]
				draw_scatter_df = self.pattern_data_df[[variable_name,self.agg_alias]]
				self.plotter.plot_2D_linear(draw_line_df,slope=slope_value,intercept=intercept_value)
				self.plotter.plot_2D_scatter(draw_scatter_df,x=variable_name,y=self.agg_alias)
				self.plotter.set_x_label(variable_name)
				self.plotter.set_y_label(self.agg_alias)
				self.plotter.set_title("pattern graph")

		canvas.draw()


	def load_pattern_description(self):

		fixed_attribute = self.chosen_row['fixed']
		fixed_value = self.chosen_row['fixed_value']
		if(len(fixed_attribute)==1):
			fixed_clause=fixed_attribute[0]+' = '+fixed_value[0]
		else:
			pairs = []
			for n in range(len(fixed_attribute)):
				pair = str(fixed_attribute[n])+' = '+str(fixed_value[n])
				pairs.append(pair)
			fixed_clause=','.join(pairs)
		aggregation_function=self.chosen_row['agg']
		modeltype = self.chosen_row['model']
		variable_attribute = self.chosen_row['variable']
		if(len(variable_attribute)==1):
			variable_attribute=variable_attribute[0]
		else:
			variable_attribute=','.join(variable_attribute)
		if(self.chosen_row['model']=='const'):
			pass
			model_str = "\n"
		else:
			Intercept_value = round((self.chosen_row['param']['Intercept']),2)
			slope_name = list(self.chosen_row['param'])[1]
			slope_value = round((self.chosen_row['param'][slope_name]),2)
			model_str = "\n\nIntercept: "+str(Intercept_value)+', '+str(slope_name)+" as Coefficient: "+str(slope_value)
		theta = "\n\nThe goodness of fit of the model is "+str(round(self.chosen_row['theta'],2))
		local_desc = "For "+fixed_clause+',\n\nthe '+aggregation_function +' is '+modeltype+' in '+variable_attribute+'.'
		local_desc = local_desc.replace('const','constant')
		pattern_attr = model_str+theta
		pattern_description = Label(self.win_frame,text=local_desc+pattern_attr,font=('Times New Roman bold',18),borderwidth=5,relief=SOLID,justify=LEFT)
		pattern_description.grid(column=0,row=0,sticky='nsew')


