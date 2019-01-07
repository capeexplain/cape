import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.messagebox
from tkinter import filedialog
from tkinter import font
import pandas as pd
import psycopg2
from tkintertable import TableCanvas, TableModel,Plot
import tkinter.colorchooser
from pandastable import PlotViewer


import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 
from matplotlib.figure import Figure
from matplotlib import style
 

conn = psycopg2.connect(dbname="antiprov",user="antiprov",host="127.0.0.1",port="5436")

# "dbname = math564 user = chenjie password = lcj53242 \
# 						host = newbballserver.ctkmtyhjwqb1.us-east-2.rds.amazonaws.com"
cur = conn.cursor() # activate cursor


test_frame = pd.read_csv('/home/chenjie/Desktop/team_stats.csv')

class CAPE_UI:

	def __init__(self,parent):

#----------------------------main frame----------------------------------------#		
		self.parent=parent
		self.main_frame_style=ttk.Style()
		self.main_frame_style.configure('Main_Frame', background='#334353')
		self.main_frame=ttk.Frame(self.parent,padding=(3,3,12,12),width=2000, height=100)

		# self.main_frame.columnconfigure(0, weight=1)
		# self.main_frame.columnconfigure(1, weight=1)
		# self.main_frame.columnconfigure(2, weight=1)

		self.main_frame.rowconfigure(0,weight=1)
		self.main_frame.rowconfigure(1,weight=1)
		self.main_frame.rowconfigure(2,weight=1)
		self.main_frame.rowconfigure(3,weight=1)
		self.main_frame.rowconfigure(4,weight=1)
		self.main_frame.rowconfigure(5,weight=1)
		self.main_frame.rowconfigure(6,weight=1)
		self.main_frame.rowconfigure(7,weight=1)
		self.main_frame.rowconfigure(8,weight=1)
		self.main_frame.rowconfigure(9,weight=1)

#----------------------------------place frames----------------------------------#
		
		self.main_frame.grid(column=0, row=0, columnspan=3, rowspan=10, sticky='nsew')

		self.table_frame = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=400,height=60)
		self.table_frame.grid(column=0,row=0, columnspan=1, rowspan=10, sticky='nsew')

		self.query_frame = ttk.Frame(self.main_frame, borderwidth=5, relief="ridge",width=600)
		self.query_frame.grid(column=1, row=0, columnspan=1, rowspan=2, sticky='nsew')

		self.query_result = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=600)
		self.query_result.grid(column=1, row=2, columnspan=1, rowspan=8, sticky='nsew')

		self.plot = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=900)
		self.plot.grid(column=2, row=0, columnspan=1, rowspan=5, sticky='nsew')

		self.explaination = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=900)
		self.explaination.grid(column=2, row=5, columnspan=1, rowspan=5, sticky='nsew')


#---------------------------table frame-----------------------------------------#
		self.table_view = ttk.Treeview(self.table_frame,height=46)
		self.table_info = Label(self.table_frame, text="Database Information")
		self.table_info.grid(column=0, row=0)
		self.table_view.grid(column=0, row=1)
		self.all_tables_query = """
							  SELECT table_name
							  FROM information_schema.tables
							  WHERE table_schema='public'
							  AND table_type='BASE TABLE';
							"""
		self.df_tables = pd.read_sql(self.all_tables_query, conn)
		
		table_index = 0
		for index, row in self.df_tables.iterrows(): 
		   self.table_view.insert('', 'end','item'+str(table_index),text = row['table_name'])
		   table_index +=1

		parent_index = 0

		for index, row in self.df_tables.iterrows():
			q = 'select column_name,data_type from information_schema.columns where table_name = ' + '\''+row['table_name'] +'\''
			table_name_attr = pd.read_sql(q,conn)
			for index, row in table_name_attr.iterrows():
				self.table_view.insert('item'+str(parent_index),'end',text=row['column_name'],values=row['data_type'])
			parent_index +=1


#----------------------------Query frame----------------------------------------#
		self.query_frame.rowconfigure(0,weight=1)
		self.query_frame.rowconfigure(1,weight=6)
		self.query_frame.rowconfigure(2,weight=1)
		self.query_frame.columnconfigure(0,weight=8)
		self.query_frame.columnconfigure(1,weight=1)
		self.query_frame.columnconfigure(2,weight=1)

		self.query_info = Label(self.query_frame,text="Query Input")
		self.query=''
		self.query_info.grid(column=0,row=0)
		self.query_entry = Text(self.query_frame, height=8, width=80)
		self.query_entry.grid(column=0,row=1)
		self.query_button = Button(self.query_frame,text="Run",command=self.run_query)
		self.query_button.grid(column=2,row=2)

#----------------------------Query result frame --------------------------------#
		
		self.query_result.columnconfigure(0,weight=15)
		self.query_result.columnconfigure(1,weight=1)
		self.query_result.rowconfigure(0,weight=1)
		self.query_result.rowconfigure(1,weight=1)
		self.query_result.rowconfigure(2,weight=1)
		self.query_result.rowconfigure(3,weight=1)
		self.query_result.rowconfigure(4,weight=10)
		self.query_result.rowconfigure(5,weight=10)
		self.query_result.rowconfigure(6,weight=10)
		self.query_result.rowconfigure(7,weight=10)
		



		self.delta = Label(self.query_result,text=u'\u0394'+': ')
		self.delta.grid(column=1,row=0)
		self.delta_entry = Entry(self.query_result,width=4)
		self.delta_entry.grid(column=1,row=0,sticky=E)

		self.theta = Label(self.query_result,text=u'\u03B8'+': ')
		self.theta.grid(column=1,row=1)
		self.theta_entry = Entry(self.query_result,width=4)
		self.theta_entry.grid(column=1,row=1,sticky=E)

		self.sigma = Label(self.query_result,text=u'\u03B4'+': ')
		self.sigma.grid(column=1,row=2)
		self.sigma_entry = Entry(self.query_result,width=4)
		self.sigma_entry.grid(column=1,row=2,sticky=E)

		self.lambda1 = Label(self.query_result,text=u'\u03BB'+': ')
		self.lambda1.grid(column=1,row=3)
		self.lambda_entry = Entry(self.query_result,width=4)
		self.lambda_entry.grid(column=1,row=3,sticky=E)

		self.show_results = Frame(self.query_result)
		self.show_results.grid(column=0,row=0,rowspan=8,sticky='nsew')

		self.query_result_model = TableModel()
		self.query_result_table = TableCanvas(self.show_results,model=self.query_result_model)
		self.query_result_table.show()

		self.low_button = Button(self.query_result,text="Low",width=5,command=self.respond_to_low)
		self.low_button.grid(column=1,row=4,sticky="ne")
		self.high_button = Button(self.query_result,text="High",width=5,command=self.respond_to_high)
		self.high_button.grid(column=1,row=5,sticky="ne")

		self.draw_button = Button(self.query_result,text="Draw",width=5,command=self.draw_test)
		self.draw_button.grid(column=1,row=6,sticky="ne")

		self.populate_button = Button(self.query_result,text="Populate",width=5,command=self.populate_axises)
		self.populate_button.grid(column=1,row=7,sticky="ne")




#-------------------------------plot frame---------------------------------------#
		self.plot.columnconfigure(0,weight=1)
		self.plot.columnconfigure(1,weight=1)
		self.plot.columnconfigure(2,weight=50)
		self.plot.rowconfigure(0,weight=1)
		self.plot.rowconfigure(1,weight=1)
		self.plot.rowconfigure(2,weight=1)
		
		self.x_axis_label = Label(self.plot,text='X')
		self.x_axis_label.grid(row=0,column=0,sticky=E)
		self.x_axis = ttk.Combobox(self.plot,width=8)
		# self.x_axis.bind('<<ComboboxSelected>>', self.get_col_name_x)
		self.x_axis.grid(row=0,column=1,sticky=W)


		self.y_axis_label = Label(self.plot,text='Y')
		self.y_axis_label.grid(row=1,column=0,sticky=E)
		self.y_axis = ttk.Combobox(self.plot,width=8)
		# self.y_axis.bind('<<ComboboxSelected>>', self.get_col_name_y)
		self.y_axis.grid(row=1,column=1,sticky=W)

		self.z_axis_label = Label(self.plot,text='Z')
		self.z_axis_label.grid(row=2,column=0,sticky=E)
		self.z_axis = ttk.Combobox(self.plot,width=8)
		# self.z_axis.bind('<<ComboboxSelected>>', self.get_col_name_z)		
		self.z_axis.grid(row=2,column=1,sticky=W)
		
		







#---------------------------------explaination frame-----------------------------#

		



#----------------------------------Functions----------------------------------------#
	def run_query(self):
		
		self.query_result_model = TableModel()
		self.query_result_table = TableCanvas(self.show_results,model=self.query_result_model)
		self.query_result_table.show()
		query = self.query_entry.get("1.0",END)
		self.query_result_df = pd.read_sql(query,conn)
		self.query_result_df['row_number'] = self.query_result_df.index
		query_result_dict= self.query_result_df.set_index(self.query_result_df['row_number'] ).T.to_dict()
		# print(self.query_result_df)
		self.new_query_result_df = pd.DataFrame.from_dict(query_result_dict)
		# self.new_query_result_df.to_csv('test.csv')
		query_model = self.query_result_table.model
		self.query_result_model.importDict(query_result_dict)
		self.query_result_table.redraw()

	def respond_to_low(self):
		print("LOW!!!!!!!"+"rowclicked: " + str(self.query_result_table.rowclicked)+"colclicked: "+ str(self.query_result_table.colclicked))
		exp_result_model = TableModel()
		self.exp_result_table = TableCanvas(self.explaination,model=exp_result_model)
		self.exp_result_table.show()
		col_clicked = list(self.query_result_table.multiplecollist)[0]
		self.exp_result_df = self.query_result_df[self.query_result_df[self.query_result_df.columns[col_clicked]]<self.query_result_df.iloc[self.query_result_table.rowclicked][self.query_result_table.colclicked]]
		self.exp_result_df['row_number'] = self.exp_result_df.index
		self.exp_result_dict= self.exp_result_df.set_index(self.exp_result_df['row_number'] ).T.to_dict()
		# print(self.result_dict)
		exp_model = self.exp_result_table.model
		exp_model.importDict(self.exp_result_dict)
		self.exp_result_table.redraw()


	def respond_to_high(self):
		print("High!!!!!!!"+"rowclicked: " + str(self.query_result_table.rowclicked)+" colclicked: "+ str(self.query_result_table.colclicked))
		exp_result_model = TableModel()
		self.exp_result_table = TableCanvas(self.explaination,model=exp_result_model)
		self.exp_result_table.show()
		col_clicked = list(self.query_result_table.multiplecollist)[0]
		self.exp_result_df = self.query_result_df[self.query_result_df[self.query_result_df.columns[col_clicked]]>self.query_result_df.iloc[self.query_result_table.rowclicked][self.query_result_table.colclicked]]
		self.exp_result_df['row_number'] = self.exp_result_df.index
		self.exp_result_dict= self.exp_result_df.set_index(self.exp_result_df['row_number'] ).T.to_dict()
		# print(self.result_dict)
		exp_model = self.exp_result_table.model
		exp_model.importDict(self.exp_result_dict)
		self.exp_result_table.redraw()

	def draw_test(self):
		self.query_result_table.getSelectionValues() # selected list
		print(self.query_result_table.multiplerowlist)
		print(self.query_result_table.multiplecollist)
		# self.query_result_col_names = 
		# self.test_figure = Figure(figsize(5,5), dpi=100)
		# self.sub_plot = self.test_figure.add_subplot(111)
		# self.sub_plot.plot()

	def get_col_name_x(self):
		x_list=[]
		col_indexes = list(self.query_result_table.multiplecollist)
		print(col_indexes)
		col_names = []
		for n in col_indexes:
			print("n : "+str(n))
			print(self.query_result_df.columns[n])
			col_names.append(self.query_result_df.columns[n])
		# for m in col_indexes:
		# 	if (m==self.x_axis.get()):
		# 		continue
		# 	elif(m==self.y_axis.get()):
		# 		continue
		# 	else:
		# 		x_list.append(m)
		self.x_axis['values'] = col_names
	
	def get_col_name_y(self):
		y_list=[]
		col_indexes = list(self.query_result_table.multiplecollist)
		print(col_indexes)
		col_names = []
		for n in col_indexes:
			print(self.query_result_df.columns[n])
			col_names.append(self.query_result_df.columns[n])
		# for m in col_indexes:
		# 	if (m==self.x_axis.get()):
		# 		continue
		# 	elif(m==self.z_axis.get()):
		# 		continue
		# 	else:
		# 		y_list.append(m)
		self.y_axis['values'] = col_names


	def get_col_name_z(self):
		z_list=[]
		col_indexes = list(self.query_result_table.multiplecollist)
		print(col_indexes)
		col_names = []
		for n in col_indexes:
			print(self.query_result_df.columns[n])
			col_names.append(self.query_result_df.columns[n])
		# for m in col_indexes:
		# 	if (m==self.x_axis.get()):
		# 		continue
		# 	elif(m==self.y_axis.get()):
		# 		continue
		# 	else:
		# 		z_list.append(m)
		self.z_axis['values'] = col_names
	def populate_axises(self):
		self.get_col_name_x()
		self.get_col_name_y()
		self.get_col_name_z()


def main():
	root = Tk()
	root.title('CAPE')
	ui = CAPE_UI(root)
	root.mainloop()		

if __name__ == '__main__':
	main()




