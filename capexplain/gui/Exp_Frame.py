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


class Exp_Frame:

	def __init__(self,parent=None):

		self.win = Toplevel()
		self.win.geometry("%dx%d%+d%+d" % (1580, 900, 250, 125))
		self.win.wm_title("Explanation Detail")

		self.win_frame = Frame(self.win)
		self.win_frame.pack(fill=BOTH,expand=True)

	def configure_frame(self,drill_down=False): # configure the frame structure according the exp type
		if(drill_down==False):
			self.rel_figure = Figure(figsize=(5,5),dpi=130)
			self.rel_plotter = Plotter(self.rel_figure)
			self.win_frame.columnconfigure(0,weight=2)
			self.win_frame.columnconfigure(1,weight=3)
			self.win_frame.rowconfigure(0,weight=8)
			self.win_frame.rowconfigure(1,weight=1)
			self.Quit_Button = Button(self.win_frame, text="Quit",width=10, height=4, command=self.win.destroy)
			self.Quit_Button.grid(column=0,row=1) 
			self.exp_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.exp_frame.grid(column=0,row=0,sticky='nesw')
			self.rel_graph_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.rel_graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')
		else:
			self.rel_figure = Figure(figsize=(5,5),dpi=130)
			self.rel_plotter = Plotter(self.rel_figure)
			self.drill_figure = Figure(figsize=(5,5),dpi=130)
			self.drill_ploter = Plotter(self.drill_figure)
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
			self.exp_frame = Frame(self.win_frame,borderwidth=5,relief=RIDGE)
			self.exp_frame.grid(column=1,columnspan=2,row=1,rowspan=2,sticky='nesw')

	# this case user question and explanation can be put in the same graph
	# def draw_rel_graph(self):

	# def draw_drill_graph(self):

	# def generate_rel_exp(self):

	# def generate_drill_exp(self):


def main():
	root = Tk()
	root.title('CAPE')
	width, height = root.winfo_screenwidth(), root.winfo_screenheight()
	root.geometry('%dx%d+0+0' % (width,height))
	exp_frame = Exp_Frame(root)
	exp_frame.configure_frame(drill_down=True)

	root.mainloop()
	

if __name__ == '__main__':
	main()

