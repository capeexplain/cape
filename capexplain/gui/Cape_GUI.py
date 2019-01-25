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
	


agg_function = re.compile('.*(sum|max|avg|min|count).*')
group_by = re.compile("group by(.*)",re.IGNORECASE)
float_num = re.compile('\d+\.\d+')

# conn = psycopg2.connect(dbname="antiprov",user="antiprov",host="127.0.0.1",port="5436")
conn = psycopg2.connect(dbname="antiprov",user="antiprov",host="127.0.0.1",port="5432",password='1234')

cur = conn.cursor() # activate cursor



class CAPE_UI:

	def __init__(self,parent):

#----------------------------main frame----------------------------------------#        
		self.parent=parent
		self.main_frame_style=ttk.Style()
		self.main_frame_style.configure('Main_Frame', background='#334353')
		self.main_frame=ttk.Frame(self.parent,padding=(3,3,12,12))

		self.main_frame.columnconfigure(0,weight=1)
		self.main_frame.columnconfigure(1,weight=3,uniform=1)
		self.main_frame.columnconfigure(2,weight=4,uniform=1)

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

		self.table_frame = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken")
		self.table_frame.grid(column=0,row=0, columnspan=1, rowspan=10, sticky='nsew')

		self.query_frame = ttk.Frame(self.main_frame, borderwidth=5, relief="ridge")
		self.query_frame.grid(column=1, row=0, columnspan=1, rowspan=2, sticky='nsew')

		self.query_result = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken")
		self.query_result.grid(column=1, row=2, columnspan=1, rowspan=8, sticky='nsew')

		self.local_pattern = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken")
		self.local_pattern.grid(column=2, row=0, columnspan=2, rowspan=5, sticky='nsew')

		self.explanation_frame = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken")
		self.explanation_frame.grid(column=2, row=5, columnspan=2, rowspan=5, sticky='nsew')


#---------------------------table frame-----------------------------------------#
		self.table_view = ttk.Treeview(self.table_frame,height=47)
		self.table_info = ttk.Label(self.table_frame, text="Database Information",font=('Times New Roman bold',12),borderwidth=5,relief=RIDGE)
		self.table_info.grid(column=0, row=0,sticky='nsew')
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
		self.query_frame.rowconfigure(3,weight=1)
		self.query_frame.rowconfigure(4,weight=1)
		self.query_frame.columnconfigure(0,weight=6)
		self.query_frame.columnconfigure(1,weight=1)

		self.query_info = Label(self.query_frame,text="Query Input",font=('Times New Roman bold',12),borderwidth=5,relief=RIDGE)
		self.query=''
		self.query_info.grid(column=0,row=0,sticky='nsew')
		self.query_entry = Text(self.query_frame, height=8,width=55)
		self.query_entry.grid(column=0,row=1,rowspan=4)
		self.query_button = Button(self.query_frame,text="Run Query",font=('Times New Roman bold',12),command=self.run_query)
		self.query_button.grid(column=1,row=2)
		self.show_global_pattern_button = Button(self.query_frame,text="Show Global Pattern",font=('Times New Roman bold',12),command=self.show_global_pattern)
		self.show_global_pattern_button.grid(column=1,row=3)
		self.show_local_pattern_button = Button(self.query_frame,text="Show Local Pattern",font=('Times New Roman bold',12),command=self.show_local_pattern)
		self.show_local_pattern_button.grid(column=1,row=4)
		

#----------------------------Query result frame --------------------------------#
		self.query_result.columnconfigure(0,weight=6)
		self.query_result.columnconfigure(1,weight=1)
		self.query_result.rowconfigure(0,weight=1)
		self.query_result.rowconfigure(1,weight=10)
		self.query_result.rowconfigure(2,weight=10)

		self.result_label = Label(self.query_result,text='Query Result',font=('Times New Roman bold',12),borderwidth=5,relief=RIDGE)
		self.result_label.grid(row=0,column=0,sticky='nsew')

		self.high_low_frame = Frame(self.query_result)
		self.high_low_frame.grid(column=1,row=1,rowspan=2,sticky='nsew')
		
		self.high_low_frame.columnconfigure(0,weight=1)
		self.high_low_frame.rowconfigure(0,weight=1)
		self.high_low_frame.rowconfigure(1,weight=1)
		self.high_low_frame.rowconfigure(2,weight=1)
		self.high_low_frame.rowconfigure(3,weight=1)
		self.high_low_frame.rowconfigure(4,weight=1)
		self.high_low_frame.rowconfigure(5,weight=1)
		self.high_low_frame.rowconfigure(6,weight=1)
		self.high_low_frame.rowconfigure(7,weight=1)


		self.high_button = Button(self.high_low_frame,text='High',font=('Times New Roman bold',12), command=self.handle_high)
		self.high_button.grid(column=0,row=1)

		self.low_button = Button(self.high_low_frame,text='Low',font=('Times New Roman bold',12), command=self.handle_low)
		self.low_button.grid(column=0,row=2)

		self.show_results = Frame(self.query_result)
		self.show_results.grid(column=0,row=1,sticky='nsew')

		self.show_global = Frame(self.query_result)
		self.show_global.grid(column=0,row=2,sticky='nsew')

		self.query_result_table = Table(self.show_results)
		self.query_result_table.show()

#---------------------------Global Pattern Frame -----------------------------------#
		
		self.show_global.rowconfigure(0,weight=1)
		self.show_global.rowconfigure(1,weight=10)
		self.show_global.columnconfigure(0,weight=1)

		self.global_pattern_label = Label(self.show_global,text="Global Patterns")
		self.global_pattern_label.grid(column=0,row=0,sticky='nsew')

		self.show_global_patterns = Frame(self.show_global)
		self.show_global_patterns.grid(column=0,row=1,sticky='nsew')

		self.global_description_button = Button(self.high_low_frame,text='Describe\nGlobal',font=('Times New Roman bold',12),command=self.global_description)
		self.global_description_button.grid(column=0,row=5)

		self.global_pattern_filter_button = Button(self.high_low_frame,text='Filter \nLocal\n Pattern',font=('Times New Roman bold',12),command=self.use_global_filter_local)
		self.global_pattern_filter_button.grid(column=0,row=6)

		self.global_pattern_table = Table(self.show_global_patterns)
		self.global_pattern_table.show()

		raw_global_pattern_query = "select CONCAT(fixed,',',variable) as set,* from dev.pub_large_no_domain_global;"

		self.raw_global_pattern_df = pd.read_sql(raw_global_pattern_query,conn)

		self.raw_global_pattern_df = self.raw_global_pattern_df.drop(['theta','dev_pos','dev_neg'],axis=1)
		self.raw_global_pattern_df['Partition'] = self.raw_global_pattern_df['fixed'].apply(lambda x: ','.join(x))
		self.raw_global_pattern_df['Predictor'] = self.raw_global_pattern_df['variable'].apply(lambda x: ','.join(x))
		self.raw_global_pattern_df['Support'] = self.raw_global_pattern_df['lambda'].round(2)
		self.raw_global_pattern_df['agg'] = 'sum(pubcount)'


#------------------------------- local pattern frame---------------------------------------#
		self.local_pattern.rowconfigure(0,weight=1)
		self.local_pattern.rowconfigure(1,weight=20)
		self.local_pattern.rowconfigure(2,weight=1)
		self.local_pattern.columnconfigure(0,weight=1)
		self.local_pattern.columnconfigure(1,weight=1)
		self.local_pattern.columnconfigure(2,weight=1)
		self.local_pattern.columnconfigure(3,weight=1)
		self.local_pattern.columnconfigure(4,weight=1)


		self.local_show_patterns = Frame(self.local_pattern)
		self.local_show_patterns.grid(column=0,row=1,sticky='nsew')

		self.local_pattern_label = Label(self.local_pattern,text="Local Patterns",font=('Times New Roman bold',12),borderwidth=5,relief=RIDGE)
		self.local_pattern_label.grid(column=0,row=0,columnspan=5,sticky='nsew')

		self.local_pattern_filter_button = Button(self.local_pattern,text='Reset Query Output',font=('Times New Roman bold',12),command=self.reset_output)
		self.local_pattern_filter_button.grid(column=0,row=2)

		self.local_pattern_filter_button = Button(self.local_pattern,text='Filter Output',font=('Times New Roman bold',12),command=self.use_local_filter_output)
		self.local_pattern_filter_button.grid(column=2,row=2)

		self.draw_pattern_button = Button(self.local_pattern,text='Draw Pattern',font=('Times New Roman bold',12),command=self.pop_up_pattern)
		self.draw_pattern_button.grid(column=4,row=2)


		self.local_pattern_table_frame = Frame(self.local_pattern)
		self.local_pattern_table_frame.grid(row=1,column=0,columnspan=5,sticky='nsew')
		self.local_pattern_table = Table(self.local_pattern_table_frame)
		self.local_pattern_table.show()

		raw_local_pattern_query = "select CONCAT(fixed,',',variable) as set, * from dev.pub_large_no_domain_local;"

		self.raw_local_pattern_df = pd.read_sql(raw_local_pattern_query,conn)

		# self.raw_local_pattern_df = self.raw_local_pattern_df.drop(['theta','dev_pos','dev_neg'],axis=1)

		self.raw_local_pattern_df['Partition'] = self.raw_local_pattern_df['fixed'].apply(lambda x: ','.join(x))
		self.raw_local_pattern_df['Partition_Values'] = self.raw_local_pattern_df['fixed_value'].apply(lambda x: ','.join(x))
		self.raw_local_pattern_df['Predictor'] = self.raw_local_pattern_df['variable'].apply(lambda x: ','.join(x))
		self.raw_local_pattern_df['stats'] = self.raw_local_pattern_df['stats'].str.split(',',expand=True)[0]
		self.raw_local_pattern_df['stats'] = self.raw_local_pattern_df['stats'].str.strip('[')
		self.raw_local_pattern_df["stats"] = pd.to_numeric(self.raw_local_pattern_df["stats"])
		self.raw_local_pattern_df["stats"] = self.raw_local_pattern_df["stats"].round(2)
		self.raw_local_pattern_df["agg"] = "sum(pubcount)"



		print(self.raw_local_pattern_df.dtypes)





#---------------------------------explanation frame-----------------------------# 
		self.explanation_frame.rowconfigure(0,weight=1)
		self.explanation_frame.rowconfigure(1,weight=10)
		self.explanation_frame.rowconfigure(2,weight=1)
		self.explanation_frame.columnconfigure(0,weight=10)
		self.exp_label = Label(self.explanation_frame,text="Top Explanations",font=('Times New Roman bold',12),borderwidth=5,relief=RIDGE)
		self.exp_label.grid(column=0,row=0,sticky='nsew')
		self.exp_table_frame = Frame(self.explanation_frame)
		self.exp_table_frame.grid(row=1,column=0,sticky='nsew')
		self.exp_table = Table(self.exp_table_frame)
		self.exp_table.show()
		self.describe_exp_button = Button(self.explanation_frame,text="Describe Explanation",font=('Times New Roman bold',12),command=self.pop_up_explanation)
		self.describe_exp_button.grid(row=2,column=0)

#----------------------------------Functions----------------------------------------#
	
	def delete_parenthesis(self,colstr):

		string1=str(colstr.replace("{","").replace("}",""))
		list1=string1.split(',')
		list1.sort()

		return  (','.join(list1))

	def run_query(self):
		
		# self.query_result_table = Table(self.show_results)
		# self.query_result_table.show()
		self.query = self.query_entry.get("1.0",END)
		print(self.query)
		self.original_query_result_df = pd.read_sql(self.query,conn)
		self.query_result_df = self.original_query_result_df
		model = TableModel(dataframe=self.original_query_result_df)
		self.query_result_table.updateModel(model)
		self.query_result_table.redraw()

	def show_global_pattern(self):

		query_list = self.query.split('\n')
		# n=0
		# for line in query_list:
		#   print(n)
		#   print(line)
		#   n+=1


		self.global_pattern_df = self.raw_global_pattern_df
		self.global_pattern_df['set'] = self.global_pattern_df['set'].apply(self.delete_parenthesis)

		query_agg = None
		query_group_list = []
		query_group_set = []
		for line in query_list:
			if(agg_function.search(line) is not None):
				query_agg = agg_function.search(line).group(1)
			if(group_by.search(line) is not None):
				query_group_list = group_by.search(line).group(1).split(',')
		for n in query_group_list: # delete whitespaces
			n = n.strip()
			query_group_set.append(n)
		query_group_set.sort()
		query_group_str = ','.join(query_group_set)
		print(query_group_str)
		print('the type is :')
		print(type(self.global_pattern_df.set.str.split(',')))

		self.global_pattern_df = self.global_pattern_df[self.global_pattern_df.apply(lambda row: query_group_str==row.set,axis=1)]

		# for index, row in self.global_pattern_df.iterrows():
		# 	row_list = row['set'].split(',')
		# 	for n in row_list:
		# 		if n not in query_group_set:
		# 			self.global_pattern_df = self.global_pattern_df.drop(index)
		
		self.global_output_pattern_df = self.global_pattern_df[['Partition','Predictor','agg','model','Support']]

		pattern_model = TableModel(dataframe=self.global_output_pattern_df)
		self.global_pattern_table.updateModel(pattern_model)
		self.global_pattern_table.redraw()

	def show_local_pattern(self):

		query_list = self.query.split('\n')
		# n=0
		# for line in query_list:
		#   print(n)
		#   print(line)
		#   n+=1


		self.local_pattern_df = self.raw_local_pattern_df
		self.local_pattern_df['set'] = self.local_pattern_df['set'].apply(self.delete_parenthesis)

		query_agg = None
		query_group_list = []
		query_group_set = []
		for line in query_list:
			if(agg_function.search(line) is not None):
				query_agg = agg_function.search(line).group(1)
			if(group_by.search(line) is not None):
				query_group_list = group_by.search(line).group(1).split(',')
		for n in query_group_list: # delete whitespaces
			n = n.strip()
			query_group_set.append(n)
		query_group_set.sort()
		query_group_str = ','.join(query_group_set)
		print(query_group_str)

		self.local_pattern_df = self.local_pattern_df[self.local_pattern_df.apply(lambda row: query_group_str==row.set,axis=1)]
		self.local_output_pattern_df = self.local_pattern_df.drop(columns=['set'],axis=1)

		local_shown = self.local_output_pattern_df[['Partition','Partition_Values','Predictor','agg']]

		pattern_model = TableModel(local_shown)
		self.local_pattern_table.updateModel(pattern_model)
		self.local_pattern_table.redraw()

	def use_global_filter_local(self):

		original_local_df = pd.DataFrame.copy(self.raw_local_pattern_df)

		print(self.global_pattern_table.multiplerowlist)

		pattern_df_lists = []

		for n in self.global_pattern_table.multiplerowlist:

			model_name = self.global_pattern_df.iloc[int(n)]['model']
			global_pattern_fixed = self.global_pattern_df.iloc[int(n)]['fixed']
			if(len(global_pattern_fixed)==1):
				global_pattern_fixed=global_pattern_fixed[0]
			else:
				global_pattern_fixed=','.join(global_pattern_fixed)
			print(global_pattern_fixed)

			global_pattern_variable = self.global_pattern_df.iloc[int(n)]['variable']
			if(len(global_pattern_variable)==1):
				global_pattern_variable=global_pattern_variable[0]
			else:
				global_pattern_variable=','.join(global_pattern_variable)
			print(global_pattern_variable)

			pattern_tuples = [[model_name,global_pattern_fixed,global_pattern_variable]]

			print(pattern_tuples)

			df = pd.DataFrame(pattern_tuples, columns=['model','fixed','variable'])
			# print(df)
			pattern_df_lists.append(df)

		filtered_df = pd.DataFrame(columns=list(self.raw_local_pattern_df))
		
		for pattern_df in pattern_df_lists:
			model=pattern_df['model'].to_string(index=False)
			print(pattern_df)
			g_fixed=pattern_df['fixed'].to_string(index=False)
			print(g_fixed)
			g_variable=pattern_df['variable'].to_string(index=False)
			print(g_variable)
			
			Q1 ="WITH model_matched as (SELECT * FROM dev.pub_large_no_domain_local WHERE model=\'"+model+"\')"+\
			"SELECT * FROM model_matched WHERE array_to_string(fixed, ',')=\'"+g_fixed+"\'AND array_to_string(variable, ',')=\'"+g_variable+'\';'
			print(Q1)
			l_result = pd.read_sql(Q1, conn)
			filtered_df = filtered_df.append(l_result,ignore_index=True)

		filtered_df['Partition'] = filtered_df['fixed'].apply(lambda x: ','.join(x))
		filtered_df['Partition_Values'] = filtered_df['fixed_value'].apply(lambda x: ','.join(x))
		filtered_df['Predictor'] = filtered_df['variable'].apply(lambda x: ','.join(x))
		filtered_df['stats'] = filtered_df['stats'].str.split(',',expand=True)[0]
		filtered_df['stats'] = filtered_df['stats'].str.strip('[')
		filtered_df["stats"] = pd.to_numeric(filtered_df["stats"])
		filtered_df["stats"] = filtered_df["stats"].round(2)
		filtered_df["agg"] = "sum(pubcount)"

		self.local_output_pattern_df  = filtered_df

		model = TableModel(dataframe=filtered_df[['Partition','Partition_Values','Predictor','agg']])
		self.local_pattern_table.updateModel(model)
		self.local_pattern_table.redraw()

	def global_description(self):

		for n in self.global_pattern_table.multiplerowlist:

			fixed_attribute = self.global_pattern_df.iloc[int(n)]['fixed']
			if(len(fixed_attribute)==1):
				fixed_attribute=fixed_attribute[0]
			else:
				fixed_attribute=','.join(fixed_attribute)
			aggregation_function=self.global_pattern_df.iloc[int(n)]['agg']
			modeltype = self.global_pattern_df.iloc[int(n)]['model']
			variable_attribute = self.global_pattern_df.iloc[int(n)]['variable']
			if(len(variable_attribute)==1):
				variable_attribute=variable_attribute[0]
			else:
				variable_attribute=','.join(variable_attribute)
			Lambda = round(self.global_pattern_df.iloc[int(n)]['lambda'],2)

		global_desc = "For each "+fixed_attribute+',\n the '+aggregation_function +' is '+modeltype+'\n in '+variable_attribute+'.'+\
		'This pattern holds for '+str(Lambda*100)+ ' % of the '+fixed_attribute
		
		desc_win = Toplevel()
		x = self.parent.winfo_x()
		print("x=" +str(x))
		y = self.parent.winfo_y()
		print("y=" +str(y))
		w = 540
		print("w=" +str(w))
		h = 120 
		print("h=" +str(h))
		desc_win.geometry("%dx%d+%d+%d" % (w, h, x + 450, y + 500))

		desc_win.wm_title("Global Pattern Description")
		desc_frame = Frame(desc_win)
		desc_frame.pack(fill=BOTH,expand=True)

		desc_label= Label(desc_frame,text=global_desc,font=('Times New Roman bold',12),borderwidth=5,relief=SOLID)
		desc_label.pack(fill=BOTH,expand=True)


	def use_local_filter_output(self):

		original_output_df = pd.DataFrame.copy(self.original_query_result_df)

		print(self.local_pattern_table.multiplerowlist)
		# print(self.pattern_table.multiplecollist)

		pattern_df_lists = []
		for n in self.local_pattern_table.multiplerowlist:

			self.chosen_local_pattern = self.local_output_pattern_df.iloc[int(n)]
			
			pattern_fixed = self.local_output_pattern_df.iloc[int(n)]['fixed']
			print(pattern_fixed)

			pattern_fixed_value = self.local_output_pattern_df.iloc[int(n)]['fixed_value']
			print(pattern_fixed_value)

			pattern_tuples = list(zip(pattern_fixed, pattern_fixed_value))

			df = pd.DataFrame(pattern_tuples, columns=['fixed','fixed_value'])
			# print(df)
			pattern_df_lists.append(df)

		
		user_query_view = 'WITH user_query as ('+self.query+')' 
		filtered_result_df = pd.DataFrame(columns=list(self.query_result_df))

		for pattern_df in pattern_df_lists:
			query_list = []
			for m in range(len(pattern_df['fixed'])):
				fixed_col_name = pattern_fixed[m]
				fixed_col_value = pattern_fixed_value[m]
				if(fixed_col_name=='year'):
					q = "SELECT * FROM user_query uq where "+fixed_col_name+"=CAST( "+fixed_col_value+" AS INT)"
				else:		
					q = "SELECT * FROM user_query uq where "+fixed_col_name+"=\'"+fixed_col_value+"\'"
				query_list.append(q)
			querybody = '\nINTERSECT\n'.join(query_list)
			full_query = user_query_view+querybody
			print(full_query)
			one_df = pd.read_sql(full_query,conn)
			print(one_df)
			filtered_result_df = filtered_result_df.append(one_df,ignore_index=True)
		
		print(filtered_result_df)
		self.query_result_df = filtered_result_df

		model = TableModel(dataframe=filtered_result_df)
		self.query_result_table.updateModel(model)
		self.query_result_table.redraw()

	def handle_low(self):
		
		self.question_tuple = ''
		config=ExplConfig()
		eg = ExplanationGenerator(config, None)
		eg.initialize() 
		col_name = ['Explanation_Tuple',"Score",'From_Pattern',"Drill_Down_To","Distance","Outlierness","Denominator"]
		exp_df = pd.DataFrame(columns=['From_Pattern',"Drill_Down_To","Score","Distance","Outlierness","Denominator"])
		for n in self.query_result_table.multiplerowlist:
			self.question = self.query_result_df.iloc[int(n)]
			self.question_tuple = self.query_result_df.iloc[[int(n)]]
			print(self.question_tuple)
			self.question['direction']='low'
			self.question['lambda'] = 0.2
			question = self.question.to_dict()
			print(question)
			elist = eg.do_explain_online(question)

			exp_list=[]
			for e in elist:
				tuple_list=[]
				e_tuple_str = ','.join(map(str, e.tuple_value.values()))
				tuple_list.append(e_tuple_str)

				score = round(e.score,2)
				tuple_list.append(score)

				if e.expl_type == 1:
					local_pattern=(
						'Partition=' + ','.join(e.relevent_pattern[0]) +'|'+\
						'Partition_Value=' + ','.join(list(map(str, e.relevent_pattern[1]))) +'|'+ \
						'Predictor=' + ','.join(list(map(str, e.relevent_pattern[2])))+'|'+\
						'Model=' + e.relevent_pattern[4] + '|'+\
						'Model Param: '
						)
					if e.relevent_pattern[4] == 'const':
						model_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
						local_pattern_full = local_pattern + model_param
					else:
						model_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(round(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]],2))
						local_pattern_full = local_pattern + model_param

					drill_down_To=(
						'Partition=' + ','.join(e.refinement_pattern[0]) +'|'+\
						'Partition_Value=' + ','.join(list(map(str, e.refinement_pattern[1]))) +'|'+ \
						'Predictor=' + ','.join(list(map(str, e.refinement_pattern[2])))+'|'+\
						'Model=' + e.refinement_pattern[4] + '|'+\
						'Model Param: '
						)
					if e.refinement_pattern[4] == 'const':
						model_param =str(round(float(e.refinement_pattern[6].split(',')[0][1:]),2))
						drill_down_To_full = drill_down_To + model_param
					else:
						model_param = 'Intercept=' + str(round(e.refinement_pattern[7]['Intercept'],2))+', '+str(list(e.refinement_pattern[7])[1])+'='+str(round(e.refinement_pattern[7][list(e.refinement_pattern[7])[1]],2))
						drill_down_To_full = drill_down_To + model_param
				else:
					local_pattern=(
						'Partition=' + ','.join(e.relevent_pattern[0]) +'|'+\
						'Partition_Value=' + ','.join(list(map(str, e.relevent_pattern[1]))) +'|'+ \
						'Predictor=' + ','.join(list(map(str, e.relevent_pattern[2])))+'|'+\
						'Model=' + e.relevent_pattern[4] + '|'+\
						'Model Param: '
						)
					if e.relevent_pattern[4] == 'const':
						model_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
						local_pattern_full = local_pattern + model_param
					else:
						model_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(round(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]],2))
						local_pattern_full = local_pattern + model_param

					drill_down_To_full = ' '

				tuple_list.append(local_pattern_full)
				tuple_list.append(drill_down_To_full)
				distance = round(e.distance,2)
				tuple_list.append(distance)
				outlierness = round(e.deviation,2)
				tuple_list.append(outlierness)
				denominator = round(e.denominator,2)
				tuple_list.append(denominator)
				exp_list.append(tuple_list)

			df_exp = pd.DataFrame(exp_list,columns=col_name)
			exp_df = exp_df.append(df_exp,ignore_index=True)
			
		self.exp_df = exp_df[col_name]
		model = TableModel(dataframe=self.exp_df)
		self.exp_table.updateModel(model)	
		self.exp_table.redraw()



	def handle_high(self):

		self.question_tuple = ''
		config=ExplConfig()
		eg = ExplanationGenerator(config, None)
		eg.initialize() 
		col_name = ['Explanation_Tuple',"Score",'From_Pattern',"Drill_Down_To","Distance","Outlierness","Denominator"]
		exp_df = pd.DataFrame(columns=['From_Pattern',"Drill_Down_To","Score","Distance","Outlierness","Denominator"])
		for n in self.query_result_table.multiplerowlist:
			self.question = self.query_result_df.iloc[int(n)]
			self.question_tuple = self.query_result_df.iloc[[int(n)]]
			print(self.question_tuple)
			self.question['direction']='high'
			self.question['lambda'] = 0.2
			question = self.question.to_dict()
			print(question)
			elist = eg.do_explain_online(question)

			exp_list=[]
			for e in elist:
				tuple_list=[]
				e_tuple_str = ','.join(map(str, e.tuple_value.values()))
				tuple_list.append(e_tuple_str)

				score = round(e.score,2)
				tuple_list.append(score)


				if e.expl_type == 1:
					local_pattern=(
						'Partition=' + ','.join(e.relevent_pattern[0]) +'|'+\
						'Partition_Value=' + ','.join(list(map(str, e.relevent_pattern[1]))) +'|'+ \
						'Predictor=' + ','.join(list(map(str, e.relevent_pattern[2])))+'|'+\
						'Model=' + e.relevent_pattern[4] + '|'+\
						'Model Param: '
						)
					if e.relevent_pattern[4] == 'const':
						model_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
						local_pattern_full = local_pattern + model_param
					else:
						model_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(round(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]],2))
						local_pattern_full = local_pattern + model_param

					drill_down_To=(
						'Partition=' + ','.join(e.refinement_pattern[0]) +'|'+\
						'Partition_Value=' + ','.join(list(map(str, e.refinement_pattern[1]))) +'|'+ \
						'Predictor=' + ','.join(list(map(str, e.refinement_pattern[2])))+'|'+\
						'Model=' + e.refinement_pattern[4] + '|'+\
						'Model Param: '
						)
					if e.refinement_pattern[4] == 'const':
						model_param =str(round(float(e.refinement_pattern[6].split(',')[0][1:]),2))
						drill_down_To_full = drill_down_To + model_param
					else:
						model_param = 'Intercept=' + str(round(e.refinement_pattern[7]['Intercept'],2))+','+str(list(e.refinement_pattern[7])[1])+'='+str(round(e.refinement_pattern[7][list(e.refinement_pattern[7])[1]],2))
						drill_down_To_full = drill_down_To + model_param
				else:
					local_pattern=(
						'Partition=' + ','.join(e.relevent_pattern[0]) +'|'+\
						'Partition_Value=' + ','.join(list(map(str, e.relevent_pattern[1]))) +'|'+ \
						'Predictor=' + ','.join(list(map(str, e.relevent_pattern[2])))+'|'+\
						'Model=' + e.relevent_pattern[4] + '|'+\
						'Model Param: '
						)
					if e.relevent_pattern[4] == 'const':
						model_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
						local_pattern_full = local_pattern + model_param
					else:
						model_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]])
						local_pattern_full = local_pattern + model_param

					drill_down_To_full = ' '

				tuple_list.append(local_pattern_full)
				tuple_list.append(drill_down_To_full)
				distance = round(e.distance,2)
				tuple_list.append(distance)
				outlierness = round(e.deviation,2)
				tuple_list.append(outlierness)
				denominator = round(e.denominator,2)
				tuple_list.append(denominator)
				exp_list.append(tuple_list)

			df_exp = pd.DataFrame(exp_list,columns=col_name)
			exp_df = exp_df.append(df_exp,ignore_index=True)
		
		self.exp_df = exp_df[col_name]
		model = TableModel(dataframe=self.exp_df)
		self.exp_table.updateModel(model)	
		self.exp_table.redraw()

	def reset_output(self):

		model = TableModel(dataframe=self.original_query_result_df)
		self.query_result_table.updateModel(model)	
		self.query_result_table.redraw()

		self.query_result_df = self.original_query_result_df


	def pop_up_pattern(self):
		win = Toplevel()
		win.geometry("%dx%d%+d%+d" % (1300, 600, 250, 125))
		win.wm_title("Window")

		win_frame = Frame(win)
		win_frame.pack(fill=BOTH,expand=True)
		win_frame.columnconfigure(0,weight=1)
		win_frame.columnconfigure(1,weight=1)
		win_frame.rowconfigure(0,weight=4)
		win_frame.rowconfigure(1,weight=1)

		for n in self.local_pattern_table.multiplerowlist:

			self.chosen_local_pattern = self.local_output_pattern_df.iloc[int(n)]

		fixed_attribute = self.chosen_local_pattern['fixed']
		fixed_value = self.chosen_local_pattern['fixed_value']
		
		if(len(fixed_attribute)==1):
			fixed_clause=fixed_attribute[0]+' = '+fixed_value[0]
		else:
			pairs = []
			for n in range(len(fixed_attribute)):
				pair = str(fixed_attribute[n])+' = '+str(fixed_value[n])
				pairs.append(pair)
			fixed_clause=','.join(pairs)

		aggregation_function=self.chosen_local_pattern['agg']
		modeltype = self.chosen_local_pattern['model']
		variable_attribute = self.chosen_local_pattern['variable']
		if(len(variable_attribute)==1):
			variable_attribute=variable_attribute[0]
		else:
			variable_attribute=','.join(variable_attribute)

		local_desc = "For "+fixed_clause+',\n\nthe '+aggregation_function +' is '+modeltype+' in '+variable_attribute+'.'
		local_desc = local_desc.replace('const','constant')

		n=self.local_pattern_table.multiplerowlist[0]
		chosen_row = self.local_output_pattern_df.iloc[int(n)]
		chosen_row = chosen_row.to_frame().T.reset_index()

		if(chosen_row['model'].to_string(index=False)=='const'):
			model_str = "\n\nConstant: "+str(round(chosen_row['stats'].values[0],2))
		else:

			Intercept_value = round((chosen_row['param'][0]['Intercept']),2)
			slope_name = list(chosen_row['param'][0])[1]
			slope_value = round((chosen_row['param'][0][slope_name]),2)

			model_str = "\n\nIntercept: "+str(Intercept_value)+', '+str(slope_name)+" as Coefficient: "+str(slope_value)

		# dev_neg = "\n\nMax Dev_Neg: "+str(round(chosen_row['dev_neg'].values[0],2))
		# dev_pos = "\n\nMax Dev_Pos: "+str(round(chosen_row['dev_pos'].values[0],2))
		theta = "\n\nThe goodness of fit of the model is "+str(round(chosen_row['theta'].values[0],2))

		pattern_attr = model_str+theta

		pattern_description = Label(win_frame,text=local_desc+pattern_attr,font=('Times New Roman bold',18),borderwidth=5,relief=SOLID,justify=LEFT)
		pattern_description.grid(column=0,row=0,sticky='nsew')

		b = ttk.Button(win_frame, text="Quit", command=win.destroy)
		b.grid(column=0,row=1,sticky='nsew')


		graph_frame = Frame(win_frame)
		graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')

		f = Figure(figsize=(10,10),dpi=130)
		a = f.add_subplot(111)

		chosen_row['variable_str'] = chosen_row['variable'].apply(lambda x: ','.join(x))
		chosen_row['fixed_str'] = chosen_row['fixed'].apply(lambda x: ','.join(x))
		variable_name = chosen_row["variable_str"].to_string(index=False)
		fixed_name = chosen_row['fixed_str'].to_string(index=False)

		pattern_data_query = "SELECT sum(pubcount) as sum_pubcount,"+variable_name+','+fixed_name+\
		"\nFROM pub_large_no_domain\nGROUP BY "+variable_name+','+fixed_name

		user_query_view = 'WITH pattern_query as ('+pattern_data_query+')' 
		pattern_data_df = pd.DataFrame(columns=list(self.query_result_df))

		query_list = []
		for m in range(len(chosen_row['fixed'][0])):
			fixed_col_name = chosen_row['fixed'][0][m]
			fixed_col_value = chosen_row['fixed_value'][0][m]
			if(fixed_col_name=='year'):
				q = "SELECT * FROM pattern_query uq where "+fixed_col_name+"=CAST( "+fixed_col_value+" AS INT)"
			else:		
				q = "SELECT * FROM pattern_query uq where "+fixed_col_name+"=\'"+fixed_col_value+"\'"
			query_list.append(q)
		querybody = '\nINTERSECT\n'.join(query_list)
		full_query = user_query_view+querybody
		print(full_query)
		one_df = pd.read_sql(full_query,conn)
		print(one_df)
		pattern_data_df = pattern_data_df.append(one_df,ignore_index=True)

		if(chosen_row['model'].to_string(index=False)=='const'):

			if(len(chosen_row['variable'][0])==1):

				f = Figure(figsize=(5,5),dpi=130)
				a = f.add_subplot(111)
				a.axhline(y=round(chosen_row['stats'].values[0],2),c="red",linewidth=2,label='Model'+str(round(chosen_row['stats'].values[0],2)))
				print(round(chosen_row['stats'].values[0],2))
				a.set_title("pattern graph")
				a.set_xlabel('Variable')
				a.set_ylabel('sum(pubcount)')
				a.legend(loc='best')
				chosen_row['variable'] = chosen_row['variable'].apply(lambda x: ','.join(x))
				variable_name = chosen_row["variable"].to_string(index=False)
				print('variable_name is ')
				print(variable_name)
				print("variable type is:")
				print(str(pattern_data_df[variable_name].dtype))

				Xuniques, X = np.unique(pattern_data_df[variable_name], return_inverse=True)
				Y = pattern_data_df['sum_pubcount']
				a.scatter(X, Y, s=60, c='b',label='sum(pubcount)')
				a.legend(loc='best')
				a.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)

			else:
				f = Figure(figsize=(5,5),dpi=130)
				a = f.gca(projection='3d')
				a.set_title("pattern graph")

				x_name = chosen_row['variable'][0][0]
				y_name = chosen_row['variable'][0][1]

				Xuniques, X = np.unique(pattern_data_df[x_name], return_inverse=True)
				print("X:")
				print(X)
				Yuniques, Y = np.unique(pattern_data_df[y_name], return_inverse=True)
				print("Y:")
				print(Y)
				# variable_1 = chosen_row["variable"][0].to_string(index=False)
				x=np.arange(X.min(),X.max()+1)
				y=np.arange(Y.min(),Y.max()+1)
				X1, Y1 = np.meshgrid(x, y)
				zs = np.array([chosen_row['stats'] for x,y in zip(np.ravel(X1), np.ravel(Y1))])
				Z = zs.reshape(X1.shape)
				a.plot_surface(X1, Y1, Z,color='r',label='Model')

				a.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)
				a.set(yticks=range(len(Yuniques)), yticklabels=Yuniques) 

				a.scatter(X, Y, pattern_data_df['sum_pubcount'],s=20, c='b',label='sum(pubcount)')
				a.set_xlabel(x_name)
				a.set_ylabel(y_name)
				a.set_zlabel('sum(pubcount)')
				# a.legend(loc='best')

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

			# toolbar = NavigationToolbar2Tk(canvas,graph_frame)
			# toolbar.update()
			# canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

		elif(chosen_row['model'].to_string(index=False)=='linear'):

			if(len(chosen_row['variable'][0])==1):

				f = Figure(figsize=(5,5),dpi=130)
				a = f.add_subplot(111)
				a.set_title("pattern graph")
				a.set_xlabel('Variable')
				a.set_ylabel('sum(pubcount)')
				a.legend(loc='best')
				chosen_row['variable'] = chosen_row['variable'].apply(lambda x: ','.join(x))
				variable_name = chosen_row["variable"].to_string(index=False)
				print('variable_name is ')
				print(variable_name)

				Intercept_value = round((chosen_row['param'][0]['Intercept']),2)
				slope_name = list(chosen_row['param'][0])[1]
				slope_value = float(chosen_row['param'][0][slope_name])
				print("slope_value is :")
				print(slope_value)

				# Xuniques, X = np.unique(pattern_data_df[variable_name], return_inverse=True)
				print('pattern_data_df variable_name min value is:')
				print(pattern_data_df[variable_name].min())
				print('pattern_data_df variable_name max value is:')
				print(pattern_data_df[variable_name].max())

				var_min = pd.to_numeric(pattern_data_df[variable_name].min())
				var_max = pd.to_numeric(pattern_data_df[variable_name].max())
				print("var_min is: ")
				print(var_min)
				print("var_max is: ")
				print(var_max)
				X1 = np.linspace(var_min-2,var_max+2,100)
				dot_min = min(X1)
				dot_max = max(X1)
				print('X1 is:')
				print(X1)
				X = pattern_data_df[variable_name]
				Y = pattern_data_df['sum_pubcount']
				y_vals = slope_value * X1 + Intercept_value
				print("y_vals are:")
				print(y_vals)
				a.plot(X1, y_vals, c='r',linewidth=2,label='Model')
				a.scatter(X, Y, s=20, c='b',label='sum(pubcount)')
				a.legend(loc='best')
				a.set_xlim([min(var_min,dot_min)-1,max(var_max,dot_max)+1])
				a.set_xlabel(variable_name)
				a.set_ylabel("sum(pubcount)")

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

		
	def pop_up_explanation(self):

		win = Toplevel()
		win.geometry("%dx%d%+d%+d" % (1300, 600, 250, 125))
		win.wm_title("Window")

		win_frame = Frame(win)
		win_frame.pack(fill=BOTH,expand=True)
		win_frame.columnconfigure(0,weight=1)
		win_frame.columnconfigure(1,weight=1)
		win_frame.rowconfigure(0,weight=4)
		win_frame.rowconfigure(1,weight=1)

		b = ttk.Button(win_frame, text="Quit", command=win.destroy)
		b.grid(column=0,row=1,sticky='nsew')

		graph_frame = Frame(win_frame)
		graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')

		f = Figure(figsize=(10,10),dpi=130)
		a = f.add_subplot(111)
		
		for n in self.exp_table.multiplerowlist:	
			relevent_pattern = self.exp_df.iloc[int(n)]['From_Pattern'].split('|')
			rel_pattern_part = relevent_pattern[0].split("=")[1]
			rel_pattern_part_value = relevent_pattern[1].split("=")[1]
			rel_pattern_pred = relevent_pattern[2].split("=")[1]
			rel_pattern_model = relevent_pattern[3].split("=")[1]
			rel_param = relevent_pattern[4].split(":")[1]
			rel_pattern_part_list = rel_pattern_part.split(',')
			rel_pattern_pred_list = rel_pattern_pred.split(',')
			rel_pattern_part_value_list = rel_pattern_part_value.split(',')
			exp_tuple = self.exp_df.iloc[int(n)]['Explanation_Tuple']
			exp_tuple_list = exp_tuple.split(',')
			exp_tuple_col = rel_pattern_part_list + rel_pattern_pred_list
			exp_tuple_col.append('pred_value')
			exp_tuple_score = float(self.exp_df.iloc[int(n)]['Score'])


			for n in range(len(exp_tuple_col)):
				if(exp_tuple_col[n]=='year' or exp_tuple_col[n]=='pred_value'):
					exp_tuple_list[n] = int(exp_tuple_list[n])
				else:
					continue

			exp_tuple_list = [exp_tuple_list]
			exp_tuple_df = pd.DataFrame(exp_tuple_list)
			print("exp_tuple_df:")
			exp_tuple_df.columns = exp_tuple_col
			print(exp_tuple_df)

		for n in range(len(rel_pattern_part_list)):
			if(rel_pattern_part_list[n]=='year'):
				continue
			else:
				rel_pattern_part_value_list[n]=('\''+rel_pattern_part_value_list[n]+'\'')

		query_pattern_value = ','.join(rel_pattern_part_value_list)

		Pattern_Q = "SELECT sum(pubcount) as sum_pubcount, "+rel_pattern_part+","+rel_pattern_pred+\
		" FROM pub_large_no_domain WHERE " + "("+rel_pattern_part+") = ("+query_pattern_value+")"+\
		" GROUP BY "+rel_pattern_pred+','+rel_pattern_part

		exp_pattern_df = pd.read_sql(Pattern_Q,conn)

		if(rel_pattern_model=='const'):

			if(len(rel_pattern_pred.split(','))==1):

				variable_name = rel_pattern_pred.split(',')[0]
				f = Figure(figsize=(5,5),dpi=130)
				a = f.add_subplot(111)
				a.axhline(y=float(rel_param),c="red",linewidth=2,label='constant = '+str(rel_param))
				a.set_title("Explanation Graph")
				a.set_xlabel('Predictor')
				a.set_ylabel('sum(pubcount)')
				a.legend(loc='best')

				Xuniques, X = np.unique(exp_pattern_df[variable_name], return_inverse=True)
				print('Xuniques:')
				print(Xuniques)
				print('X values:')
				print(X)
				Y = exp_pattern_df['sum_pubcount']
				print('Y:')
				print(Y)
				a.scatter(X, Y, s=60, c='b',label='sum(pubcount)')

				x_variable_list = exp_pattern_df[variable_name].tolist()
				print('x_variable_list:')
				print(x_variable_list)
				x_variable_list.sort()
				print("sorted x_variable_list:")
				print(x_variable_list)
				print("type in x_variable_list:")
				print(type(x_variable_list[0]))
				print("self.question_tuple[variable] is:")
				print(self.question_tuple[variable_name])

				if variable_name=='year':
					X1 = x_variable_list.index(int(self.question_tuple[variable_name]))
					X2 = x_variable_list.index(int(exp_tuple_df[variable_name]))
				else:
					X1 = x_variable_list.index(str(self.question_tuple[variable_name]))
					X2 = x_variable_list.index(str(exp_tuple_df[variable_name]))

				x=self.question_tuple[variable_name]

				if(variable_name=='year'):
					Y1 = int(exp_pattern_df.loc[exp_pattern_df[variable_name] == int(x.to_string(index=False))]['sum_pubcount'])
				else:
					Y1 = int(exp_pattern_df.loc[exp_pattern_df[variable_name] == str(x.to_string(index=False))]['sum_pubcount'])

				a.scatter(X1,Y1, s=150,marker='p',c='r',label="User Question")
				Y2 = exp_tuple_df['pred_value']
				a.scatter(X2,Y2,s=150,marker='X',c='g',label='Explanation')
				a.legend(loc='best')
				a.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)

			else:

				f = Figure(figsize=(5,5),dpi=130)
				a = f.gca(projection='3d')
				a.set_title("Explanation Graph")

				x_name = rel_pattern_pred.split(',')[0]
				y_name = rel_pattern_pred.split(',')[1]

				Xuniques, X = np.unique(exp_pattern_df[x_name], return_inverse=True)
				print("X:")
				print(X)
				Yuniques, Y = np.unique(exp_pattern_df[y_name], return_inverse=True)
				print("Y:")
				print(Y)
				# variable_1 = chosen_row["variable"][0].to_string(index=False)
				x=np.arange(X.min(),X.max()+1)
				y=np.arange(Y.min(),Y.max()+1)
				X1, Y1 = np.meshgrid(x, y)
				zs = np.array([float(rel_param) for x,y in zip(np.ravel(X1), np.ravel(Y1))])
				Z = zs.reshape(X1.shape)
				a.plot_surface(X1, Y1, Z,color='r')

				a.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)
				a.set(yticks=range(len(Yuniques)), yticklabels=Yuniques)

				x_variable_list = exp_pattern_df[x_name].tolist()
				x_variable_list.sort()
				y_variable_list = exp_pattern_df[y_name].tolist()
				y_variable_list.sort()


				if(x_name=='year'):
					X1 = x_variable_list.index(int(self.question_tuple[x_name]))
					X2 = x_variable_list.index(int(exp_tuple_df[x_name]))
				else:
					X1 = x_variable_list.index(str(self.question_tuple[x_name]))
					X2 = x_variable_list.index(str(exp_tuple_df[x_name]))

				if(y_name=='year'):
					Y1 = y_variable_list.index(int(self.question_tuple[y_name]))
					Y2 = y_variable_list.index(int(exp_tuple_df[y_name]))
				else:
					Y1 = y_variable_list.index(str(self.question_tuple[y_name]))
					Y2 = y_variable_list.index(str(exp_tuple_df[y_name]))

				a.scatter(X, Y, exp_pattern_df['sum_pubcount'],s=60, c='b',label='sum(pubcount)')
				a.scatter(X1,Y1,self.question_tuple['sum_pubcount'],s=150,marker='p',c='r',label='User Question')
				a.scatter(X2,Y2,exp_tuple_df['pred_value'],s=150,marker='X',c='g',label='Explanation')
				a.legend(loc='best')
				a.set_xlabel(x_name)
				a.set_ylabel(y_name)
				a.set_zlabel('sum(pubcount)')

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

			# toolbar = NavigationToolbar2Tk(canvas,graph_frame)
			# toolbar.update()
			# canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

		elif(rel_pattern_model=='linear'):

			if(len(rel_pattern_pred.split(','))==1):

				variable_name = rel_pattern_pred.split(',')[0]
				f = Figure(figsize=(5,5),dpi=130)
				a = f.add_subplot(111)
				a.set_title("Explanation Graph")
				a.set_xlabel('Variable')
				a.set_ylabel('sum(pubcount)')
				a.legend(loc='best')

				Intercept_value = float(rel_param.split(",")[0].split("=")[1])
				slope_name = rel_param.split(',')[1].split('=')[0].strip()
				slope_value = float(rel_param.split(',')[1].split('=')[1].strip())
				print("slope_value is :")
				print(slope_value)

				var_min = pd.to_numeric(exp_pattern_df[variable_name].min())
				var_max = pd.to_numeric(exp_pattern_df[variable_name].max())
				print("var_min is: ")
				print(var_min)
				print("var_max is: ")
				print(var_max)
				X1 = np.linspace(var_min-2,var_max+2,100)
				dot_min = min(X1)
				dot_max = max(X1)
				print('X1 is:')
				print(X1)
				X = exp_pattern_df[variable_name]
				Y = exp_pattern_df['sum_pubcount']
				X2 = self.question_tuple[variable_name]
				print('X2:')
				print(X2)
				Y2 = int(exp_pattern_df.loc[exp_pattern_df[variable_name] == int(X2.to_string(index=False))]['sum_pubcount'])
				X3 = exp_tuple_df[variable_name]
				y_vals = slope_value * X1 + Intercept_value
				print("y_vals are:")
				print(y_vals)
				a.plot(X1, y_vals, c='r',linewidth=2,label="Model")
				a.scatter(X, Y, s=60, c='b',label='sum(pubcount)')
				a.scatter(X2,Y2,s=150,marker='p',c='r',label='User Question')
				a.scatter(X3,exp_tuple_df['pred_value'],s=150,marker='X',c='g',label='Explanation')
				a.legend(loc='best')
				a.set_xlim([min(var_min,dot_min)-1,max(var_max,dot_max)+1])
				a.set_xlabel(variable_name)
				a.set_ylabel("sum(pubcount)")

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

		likelihood_words = []
		self.question_tuple['direction']='high'

		if(exp_tuple_score<=0):
			likelihood_words = ['unlikely','not similar','slighlty']
		elif(exp_tuple_score<=10):
			likelihood_words = ['plausible','somewhat similar','']
		else:
			likelihood_words = ['highly plausible','similar','extremly']

		ranking_clause = "This explanation was ranked "+ likelihood_words[0] + "\nbecause the counterbalance is " + likelihood_words[1]+\
		" to the user question\nand it deviates "+likelihood_words[2]+"from the predicted outcome.\n"

		print('ranking_clause:')
		print(ranking_clause)

		print('self.question_tuple is:')
		print(self.question_tuple)
		user_question_dict = self.question_tuple.to_dict('records')[0]
		print('user_question_dict:')
		print(user_question_dict)
		user_question_list = []
		for k,v in user_question_dict.items():
			if(k=='sum_pubcount' or k=='direction'):
				continue
			else:
				user_question_list.append(str(k)+"="+str(v))
		user_question_clause = ','.join(user_question_list)
		print("user_question_list")
		print(user_question_clause)

		predict = '' 
		if(len(rel_pattern_pred.split(','))>1):
			predict = 'predict'
		else:
			predict = 'predicts'	

		fixed_pair_list=[]
		fixed_attr_list = str(self.chosen_local_pattern['Partition']).split(',')
		fixed_value_list = str(self.chosen_local_pattern['Partition_Values']).split(',')

		for n in range(len(fixed_attr_list)):
			eq = (fixed_attr_list[n]+"="+fixed_value_list[n])
			fixed_pair_list.append(eq)
		if(len(fixed_pair_list)==1):
			fixed_pair = fixed_pair_list[0]
		else:
			fixed_pair = ",".join(fixed_pair_list)

		variable_pair_list=[]
		variable_attr_list = str(self.chosen_local_pattern['Predictor']).split(',')

		for n in range(len(variable_attr_list)):
			eq = (str(variable_attr_list[n])+"="+str(self.question_tuple[variable_attr_list[n]].to_string(index=False)))
			variable_pair_list.append(eq)
		if(len(variable_pair_list)==1):
			variable_pair = variable_pair_list[0]
		else:
			variable_pair = ",".join(variable_pair_list)

		counter_dir = ''

		if(str(self.question_tuple['direction'].to_string(index=False))=='high'):
			counter_dir='low'
		else:
			counter_dir='high'

		exp_pair = ''

		print('exp_tuple_df:')
		print(exp_tuple_df)


		exp_tuple_dict = exp_tuple_df.to_dict('records')[0]

		exp_list = []
		for k,v in exp_tuple_dict.items():
			if(k=='pred_value' or k in rel_pattern_part.split(',')):
				continue
			else:
				exp_list.append(str(k)+"="+str(v))
		exp_clause = ','.join(exp_list)


		fixed_exp_pair_list = []
		for n in range(len(rel_pattern_part_list)):
			eq = (rel_pattern_part_list[n]+"="+rel_pattern_part_value_list[n])
			fixed_exp_pair_list.append(eq)
		if(len(fixed_exp_pair_list)==1):
			fixed_pair = fixed_exp_pair_list[0]
		else:
			fixed_pair = ",".join(fixed_exp_pair_list)


		comprehensive_exp = "\n\nExplanation for why sum(pubcount) is " + self.question_tuple['direction'].to_string(index=False)+"er\nthan expected for: "+user_question_clause+\
		"\nIn general, "+str(rel_pattern_pred)+" "+predict+" sum(pubcount) for most "+str(rel_pattern_part)+"."+\
		"\nThis is also true for "+ fixed_pair+'.'\
		"\nHowever, for "+variable_pair+",\nthe value of sum(pubcount) is "+ self.question_tuple['direction'].to_string(index=False)+"er than predicted."+\
		"\nThis may be explained through the "+counter_dir+"er than expected outcome \nfor "+ exp_clause+"."

		comprehensive_exp = comprehensive_exp.replace('name','author')

		print('comprehensive_exp:')
		print(comprehensive_exp)

		pattern_description = Label(win_frame,text=ranking_clause+comprehensive_exp,font=('Times New Roman bold',13),borderwidth=5,relief=SOLID,justify=LEFT)
		pattern_description.grid(column=0,row=0,sticky='nsew')




def main():
	root = Tk()
	root.title('CAPE')
	width, height = root.winfo_screenwidth(), root.winfo_screenheight()
	root.geometry('%dx%d+0+0' % (width,height))
	ui = CAPE_UI(root)

	root.mainloop()
	

if __name__ == '__main__':
	main()




