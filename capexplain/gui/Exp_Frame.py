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



class Exp_Frame:

	def __init__(self,parent=None):

		self.win = Toplevel()
		self.win.geometry("%dx%d%+d%+d" % (1580, 700, 250, 125))
		self.win.wm_title("Explanation")

		self.win_frame = Frame(self.win)
		self.win_frame.pack(fill=BOTH,expand=True)


		self.win_frame.columnconfigure(0,weight=2)
		self.win_frame.columnconfigure(1,weight=3)
		self.win_frame.rowconfigure(0,weight=4)
		self.win_frame.rowconfigure(1,weight=1)

		b = ttk.Button(self.win_frame, text="Quit",width=10, height=6, command=self.win.destroy)
		b.grid(column=0,row=1)

		graph_frame = Frame(self.win_frame)
		graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')

		f = Figure(figsize=(10,10),dpi=130)
		a = f.add_subplot(111)

	def handle_relevent(self,relevent_exp_tuple):
		self.relevent_exp_tuple = relevent_exp_tuple

	def handle_refinement(self,refinement_exp_tuple):
		self.refinement_exp_tuple = refinement_exp_tuple



def main():
	root = Tk()
	root.title('CAPE')
	width, height = root.winfo_screenwidth(), root.winfo_screenheight()
	root.geometry('%dx%d+0+0' % (width,height))
	ui = Exp_Window(root)

	root.mainloop()
	

if __name__ == '__main__':
	main()

