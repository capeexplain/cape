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
from_which = re.compile('from (.*)',re.IGNORECASE)
agg_alias = re.compile('as\s+(\w+)',re.IGNORECASE) 
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
		for row in self.df_tables.itertuples(): 
		   self.table_view.insert('', 'end','item'+str(table_index),text = row.table_name)
		   table_index +=1

		parent_index = 0

		for row in self.df_tables.itertuples():
			q = 'select column_name,data_type from information_schema.columns where table_name = ' + '\''+row.table_name +'\''
			table_name_attr = pd.read_sql(q,conn)
			for row in table_name_attr.itertuples():
				self.table_view.insert('item'+str(parent_index),'end',text=row.column_name,values=row.data_type)
			parent_index +=1

		self.pub_dict = {"dict_name":"pub",
		"global_name":"dev.pub_global",
		"local_name":"dev.pub_local"
		}

		self.crime_dict = {"dict_name":"crime",
		"global_name": "dev.crime_global",
		"local_name":"dev.crime_local"
		}


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

		self.parent.columnconfigure(0, weight=1)
		self.parent.rowconfigure(0, weight=1)

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

		# create a sql function for sorting the array in order to be used for filtering

		sort_array_function = "CREATE OR REPLACE FUNCTION array_sort(anyarray) RETURNS anyarray AS $$"+\
		"SELECT array_agg(x order by x) FROM unnest($1) x;"+\
		"$$ LANGUAGE sql;"

		cur.execute(sort_array_function)	


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
		
		self.query = self.query_entry.get("1.0",END)
		print(self.query)
		self.handle_view ="\nDROP VIEW IF EXISTS user_query;"+\
		"\nCREATE VIEW user_query as "+ self.query
		print(self.handle_view)
		cur.execute(self.handle_view)
		self.original_query_result_df = pd.read_sql(self.query,conn)

		self.query_result_df = self.original_query_result_df
		model = TableModel(dataframe=self.original_query_result_df)

		self.query_result_table.updateModel(model)
		self.query_result_table.redraw()

		query_list = self.query.split('\n')

		query_agg = None
		table_name = None
		self.agg_name = None
		query_group_list = []
		query_group_set = []
		for line in query_list:
			if(agg_function.search(line) is not None):
				query_agg = agg_function.search(line).group(1)
			if(group_by.search(line) is not None):
				query_group_list = group_by.search(line).group(1).split(',')
		for line1 in query_list:
			if(from_which.search(line1) is not None):
				table_name = from_which.search(line1).group(1)
			else:
				continue
		for line1 in query_list:
			if(agg_alias.search(line1) is not None):
				self.agg_name = agg_alias.search(line1).group(1)
			else:
				continue

		if(table_name.lower()==self.pub_dict['dict_name']):
			self.table_dict = self.pub_dict
		elif(table_name.lower()==self.crime_dict['dict_name']):
			self.table_dict = self.crime_dict

		for n in query_group_list: # delete whitespaces
			n = n.strip()
			n = n.strip(';')
			query_group_set.append(n)
		query_group_set.sort()
		self.query_group_str = ','.join(query_group_set)

	def show_global_pattern(self):

		global_query = "select array_to_string(fixed,',') as Partition,array_to_string(variable,',') as Predictor,agg,"+\
		"round((lambda)::numeric(4,2),2) as Support,model from "+self.table_dict['global_name']+\
		" where array_to_string(array_sort(fixed||variable),',')='"+self.query_group_str+"';"
		print(global_query)
		self.global_pattern_df = pd.read_sql(global_query,conn)
		print(self.global_pattern_df.head())
		print(list(self.global_pattern_df))

		pattern_model = TableModel(dataframe=self.global_pattern_df)
		self.global_pattern_table.updateModel(pattern_model)
		self.global_pattern_table.redraw()

	def show_local_pattern(self):


		local_query = "select array_to_string(fixed,',') as Partition,array_to_string(variable,',') as Predictor,"+\
		"array_to_string(fixed_value,',') as partition_values,agg,model,fixed,fixed_value,variable,"+\
		"theta,param,stats,dev_pos,dev_neg from "+self.table_dict['local_name']+\
		" where array_to_string(array_sort(fixed||variable),',')='"+self.query_group_str+"';"

		for n in self.local_pattern_table.multiplerowlist:
			self.chosen_local_pattern = self.local_output_pattern_df.iloc[int(n)]

		self.local_output_pattern_df = pd.read_sql(local_query,conn)
		self.local_output_pattern_df = pd.read_sql(g_filter_l_query,conn)
		self.local_output_pattern_df['stats'] = self.local_output_pattern_df['stats'].str.split(',',expand=True)[0]
		self.local_output_pattern_df['stats'] = self.local_output_pattern_df['stats'].str.strip('[')
		self.local_output_pattern_df["stats"] = pd.to_numeric(self.local_output_pattern_df["stats"])
		self.local_output_pattern_df["stats"] = self.local_output_pattern_df["stats"].round(2)

		local_shown = self.local_output_pattern_df[['partition','partition_values','predictor','agg']]

		pattern_model = TableModel(local_shown)
		self.local_pattern_table.updateModel(pattern_model)
		self.local_pattern_table.redraw()

	def use_global_filter_local(self):

		pattern_df_lists = []

		for n in self.global_pattern_table.multiplerowlist:

			model_name = self.global_pattern_df.iloc[int(n)]['model']
			print("model_name"+model_name)
			global_partition = self.global_pattern_df.iloc[int(n)]['partition']
			global_predictor = self.global_pattern_df.iloc[int(n)]['predictor']

			g_filter_l_query = " select array_to_string(fixed,',') as Partition,array_to_string(variable,',') as Predictor,"+\
			"array_to_string(fixed_value,',') as partition_values,agg,model,fixed,fixed_value,variable,"+\
			"theta,param,stats,dev_pos,dev_neg from "+self.table_dict['local_name']+\
			" where array_to_string(fixed,',')='"+global_partition+\
			"' and array_to_string(variable,',')='"+global_predictor+\
			"' and model = '"+model_name+"';"

			print(g_filter_l_query)

			self.local_output_pattern_df = pd.read_sql(g_filter_l_query,conn)
			self.local_output_pattern_df['stats'] = self.local_output_pattern_df['stats'].str.split(',',expand=True)[0]
			self.local_output_pattern_df['stats'] = self.local_output_pattern_df['stats'].str.strip('[')
			self.local_output_pattern_df["stats"] = pd.to_numeric(self.local_output_pattern_df["stats"])
			self.local_output_pattern_df["stats"] = self.local_output_pattern_df["stats"].round(2)

			local_shown = self.local_output_pattern_df[['partition','partition_values','predictor','agg']]

		model = TableModel(dataframe=local_shown)
		self.local_pattern_table.updateModel(model)
		self.local_pattern_table.redraw()

	def global_description(self):

		for n in self.global_pattern_table.multiplerowlist:
			fixed_attribute = self.global_pattern_df.iloc[int(n)]['partition']
			aggregation_function=self.global_pattern_df.iloc[int(n)]['agg']
			modeltype = self.global_pattern_df.iloc[int(n)]['model']
			variable_attribute = self.global_pattern_df.iloc[int(n)]['predictor']
			Lambda = self.global_pattern_df.iloc[int(n)]['support']

		global_desc = "For each "+fixed_attribute+',the '+aggregation_function +' is '+modeltype+'\n in '+variable_attribute+'.'+\
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
		desc_label= Label(desc_frame,text=global_desc,font=('Times New Roman bold',12),borderwidth=5,relief=SOLID,justify=LEFT)
		desc_label.pack(fill=BOTH,expand=True)


	def use_local_filter_output(self):

		concat_clause = None
		l_filter_o_query = None
		fixed_list=[]
		for n in self.local_pattern_table.multiplerowlist:

			self.chosen_local_pattern = self.local_output_pattern_df.iloc[int(n)]

			fixed_str = self.local_output_pattern_df.iloc[int(n)]['partition']
			variable_str = self.local_output_pattern_df.iloc[int(n)]['predictor']
			partitions = self.local_output_pattern_df.iloc[int(n)]['partition_values']
			fixed_str_list = fixed_str.split(',')
			if(len(fixed_str_list)==1):
				concat_clause = fixed_str
			else:
				concat_clause = "||','||".join(fixed_str_list)
		
		l_filter_o_query = "select user_query.* from user_query where "+concat_clause+"='"+partitions+"';"

		filtered_result_df = pd.read_sql(l_filter_o_query,conn)

		self.query_result_df = filtered_result_df

		model = TableModel(dataframe=filtered_result_df)
		self.query_result_table.updateModel(model)
		self.query_result_table.redraw()


	def handle_low(self):
		
		self.question_tuple = ''
		config=ExplConfig()
		eg = ExplanationGenerator(config, None)
		eg.initialize() 
		col_name = ['Explanation_Tuple',"Score",'From_Pattern',"Drill_Down_To","Distance","Outlierness","Denominator","relevent_param","drill_param"]
		exp_df = pd.DataFrame(columns=["From_Pattern","Drill_Down_To","Score","Distance","Outlierness","Denominator","relevent_param","drill_param"])
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
						'[' + ','.join(e.relevent_pattern[0]) +\
						' = ' + ','.join(list(map(str, e.relevent_pattern[1]))) +'] : '+ \
						','.join(list(map(str, e.relevent_pattern[2])))+'>>'+\
						e.relevent_pattern[4] + '>>'+ self.agg_name
						)
					if e.relevent_pattern[4] == 'const':
						relevent_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
					# 	local_pattern_full = local_pattern + model_param
					else:
						relevent_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(round(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]],2))
					# 	local_pattern_full = local_pattern + model_param

					# drill_down_To=(
					# 	'Partition=' + ','.join(e.refinement_pattern[0]) +'|'+\
					# 	'Partition_Value=' + ','.join(list(map(str, e.refinement_pattern[1]))) +'|'+ \
					# 	'Predictor=' + ','.join(list(map(str, e.refinement_pattern[2])))+'|'+\
					# 	'Model=' + e.refinement_pattern[4] + '|'+\
					# 	'Model Param: '
					# 	)
					drill_down_to = ','.join([x for x in e.refinement_pattern[0] if x not in e.relevent_pattern[0]])
					if e.refinement_pattern[4] == 'const':

						drill_param =str(round(float(e.refinement_pattern[6].split(',')[0][1:]),2))
						# drill_down_To_full = drill_down_To + model_param
						drill_down_To_full = drill_down_to + '>>const>>'+self.agg_name
					else:
						# model_param = 'Intercept=' + str(round(e.refinement_pattern[7]['Intercept'],2))+','+str(list(e.refinement_pattern[7])[1])+'='+str(round(e.refinement_pattern[7][list(e.refinement_pattern[7])[1]],2))
						# drill_down_To_full = drill_down_To + model_param
						drill_down_To_full = drill_down_to + '>>linear>>' + self.agg_name
				else:
					# local_pattern=(
					# 	'Partition=' + ','.join(e.relevent_pattern[0]) +'|'+\
					# 	'Partition_Value=' + ','.join(list(map(str, e.relevent_pattern[1]))) +'|'+ \
					# 	'Predictor=' + ','.join(list(map(str, e.relevent_pattern[2])))+'|'+\
					# 	'Model=' + e.relevent_pattern[4] + '|'+\
					# 	'Model Param: '
					# 	)
					local_pattern=(
						'[' + ','.join(e.relevent_pattern[0]) +\
						' = ' + ','.join(list(map(str, e.relevent_pattern[1]))) +'] : '+ \
						','.join(list(map(str, e.relevent_pattern[2])))+'>>'+\
						e.relevent_pattern[4] + '>>'+ self.agg_name
						)
					if e.relevent_pattern[4] == 'const':
						relevent_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
					# 	local_pattern_full = local_pattern + model_param
					else:
						relevent_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(round(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]],2))
					# local_pattern_full = local_pattern + model_param

					drill_down_To_full = ''
					drill_param = ''

				tuple_list.append(local_pattern)
				tuple_list.append(drill_down_To_full)
				distance = round(e.distance,2)
				tuple_list.append(distance)
				outlierness = round(e.deviation,2)
				tuple_list.append(outlierness)
				denominator = round(e.denominator,2)
				tuple_list.append(denominator)
				tuple_list.append(relevent_param)
				tuple_list.append(drill_param)
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
		col_name = ['Explanation_Tuple',"Score",'From_Pattern',"Drill_Down_To","Distance","Outlierness","Denominator","relevent_param","drill_param"]
		exp_df = pd.DataFrame(columns=["From_Pattern","Drill_Down_To","Score","Distance","Outlierness","Denominator","relevent_param","drill_param"])
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
						'[' + ','.join(e.relevent_pattern[0]) +\
						' = ' + ','.join(list(map(str, e.relevent_pattern[1]))) +'] : '+ \
						','.join(list(map(str, e.relevent_pattern[2])))+'>>'+\
						e.relevent_pattern[4] + '>>'+ self.agg_name
						)
					if e.relevent_pattern[4] == 'const':
						relevent_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
					# 	local_pattern_full = local_pattern + model_param
					else:
						relevent_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(round(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]],2))
					# 	local_pattern_full = local_pattern + model_param

					# drill_down_To=(
					# 	'Partition=' + ','.join(e.refinement_pattern[0]) +'|'+\
					# 	'Partition_Value=' + ','.join(list(map(str, e.refinement_pattern[1]))) +'|'+ \
					# 	'Predictor=' + ','.join(list(map(str, e.refinement_pattern[2])))+'|'+\
					# 	'Model=' + e.refinement_pattern[4] + '|'+\
					# 	'Model Param: '
					# 	)
					drill_down_to = ','.join([x for x in e.refinement_pattern[0] if x not in e.relevent_pattern[0]])
					if e.refinement_pattern[4] == 'const':

						drill_param =str(round(float(e.refinement_pattern[6].split(',')[0][1:]),2))
						# drill_down_To_full = drill_down_To + model_param
						drill_down_To_full = drill_down_to + '>>const>>'+self.agg_name
					else:
						# model_param = 'Intercept=' + str(round(e.refinement_pattern[7]['Intercept'],2))+','+str(list(e.refinement_pattern[7])[1])+'='+str(round(e.refinement_pattern[7][list(e.refinement_pattern[7])[1]],2))
						# drill_down_To_full = drill_down_To + model_param
						drill_down_To_full = drill_down_to + '>>linear>>' + self.agg_name
				else:
					# local_pattern=(
					# 	'Partition=' + ','.join(e.relevent_pattern[0]) +'|'+\
					# 	'Partition_Value=' + ','.join(list(map(str, e.relevent_pattern[1]))) +'|'+ \
					# 	'Predictor=' + ','.join(list(map(str, e.relevent_pattern[2])))+'|'+\
					# 	'Model=' + e.relevent_pattern[4] + '|'+\
					# 	'Model Param: '
					# 	)
					local_pattern=(
						'[' + ','.join(e.relevent_pattern[0]) +\
						' = ' + ','.join(list(map(str, e.relevent_pattern[1]))) +'] : '+ \
						','.join(list(map(str, e.relevent_pattern[2])))+'>>'+\
						e.relevent_pattern[4] + '>>'+ self.agg_name
						)
					if e.relevent_pattern[4] == 'const':
						relevent_param = str(round(float(e.relevent_pattern[6].split(',')[0][1:]),2))
					# 	local_pattern_full = local_pattern + model_param
					else:
						relevent_param = 'Intercept=' + str(round(e.relevent_pattern[7]['Intercept'],2))+', '+str(list(e.relevent_pattern[7])[1])+'='+str(round(e.relevent_pattern[7][list(e.relevent_pattern[7])[1]]),2)
					# local_pattern_full = local_pattern + model_param

					drill_down_To_full = ''
					drill_param = ''

				tuple_list.append(local_pattern)
				tuple_list.append(drill_down_To_full)
				distance = round(e.distance,2)
				tuple_list.append(distance)
				outlierness = round(e.deviation,2)
				tuple_list.append(outlierness)
				denominator = round(e.denominator,2)
				tuple_list.append(denominator)
				tuple_list.append(relevent_param)
				tuple_list.append(drill_param)
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
		win.wm_title("Pattern")

		win_frame = Frame(win)
		win_frame.pack(fill=BOTH,expand=True)
		win_frame.columnconfigure(0,weight=1)
		win_frame.columnconfigure(1,weight=1)
		win_frame.rowconfigure(0,weight=4)
		win_frame.rowconfigure(1,weight=1)

		concat_clause = None
		l_filter_o_query = None
		fixed_list=[]
		for n in self.local_pattern_table.multiplerowlist:

			self.chosen_local_pattern = self.local_output_pattern_df.iloc[int(n)]

			fixed_str = self.local_output_pattern_df.iloc[int(n)]['partition']
			variable_str = self.local_output_pattern_df.iloc[int(n)]['predictor']
			partitions = self.local_output_pattern_df.iloc[int(n)]['partition_values']
			fixed_str_list = fixed_str.split(',')
			if(len(fixed_str_list)==1):
				concat_clause = fixed_str
			else:
				concat_clause = "||','||".join(fixed_str_list)

		chosen_row = self.chosen_local_pattern

		print("chosen_row:")
		print(chosen_row)
		print("\n")
		
		l_filter_o_query = "select user_query.* from user_query where "+concat_clause+"='"+partitions+"';"
		
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

		if(chosen_row['model']=='const'):
			pass
			model_str = "\n"
		else:

			Intercept_value = round((chosen_row['param']['Intercept']),2)
			slope_name = list(chosen_row['param'])[1]
			slope_value = round((chosen_row['param'][slope_name]),2)

			model_str = "\n\nIntercept: "+str(Intercept_value)+', '+str(slope_name)+" as Coefficient: "+str(slope_value)

		theta = "\n\nThe goodness of fit of the model is "+str(round(chosen_row['theta'],2))

		local_desc = "For "+fixed_clause+',\n\nthe '+aggregation_function +' is '+modeltype+' in '+variable_attribute+'.'
		local_desc = local_desc.replace('const','constant')

		pattern_attr = model_str+theta

		pattern_description = Label(win_frame,text=local_desc+pattern_attr,font=('Times New Roman bold',18),borderwidth=5,relief=SOLID,justify=LEFT)
		pattern_description.grid(column=0,row=0,sticky='nsew')

		b = ttk.Button(win_frame, text="Quit", command=win.destroy)
		b.grid(column=0,row=1,sticky='nsew')


		graph_frame = Frame(win_frame)
		graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')

		f = Figure(figsize=(10,10),dpi=130)
		a = f.add_subplot(111)

		pattern_data_df = pd.read_sql(l_filter_o_query,conn)

		print("pattern_data_df")
		print(pattern_data_df)
		print("len(chosen_row['variable']) is ")
		print(len(chosen_row['variable']))

		if(chosen_row['model']=='const'):

			if(len(chosen_row['variable'])==1):

				f = Figure(figsize=(5,5),dpi=130)
				a = f.add_subplot(111)
				a.axhline(y=round(chosen_row['stats'],2),c="red",linewidth=2,label='Model: '+str(round(chosen_row['stats'],2)))
				print(round(chosen_row['stats'],2))
				a.set_title("pattern graph")
				a.set_xlabel('Variable')
				a.set_ylabel(self.agg_name)
				a.legend(loc='best')
				variable_name= chosen_row['variable'][0]
				print('variable_name is ')
				print(variable_name)
				print("variable type is:")
				print(str(pattern_data_df[variable_name].dtype))

				Xuniques, X = np.unique(pattern_data_df[variable_name], return_inverse=True)
				Y = pattern_data_df[self.agg_name]
				a.scatter(X, Y, s=60, c='b',label=self.agg_name)
				a.legend(loc='best')
				a.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)

			else:
				f = Figure(figsize=(5,5),dpi=130)
				a = f.gca(projection='3d')
				a.set_title("pattern graph")

				x_name = chosen_row['variable'][0]
				y_name = chosen_row['variable'][1]

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

				a.scatter(X, Y, pattern_data_df[self.agg_name],s=20, c='b',label=self.agg_name)
				a.set_xlabel(x_name)
				a.set_ylabel(y_name)
				a.set_zlabel(self.agg_name)

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

		elif(chosen_row['model']=='linear'):

			if(len(chosen_row['variable'])==1):

				f = Figure(figsize=(5,5),dpi=130)
				a = f.add_subplot(111)
				a.set_title("pattern graph")
				a.set_xlabel('Variable')
				a.set_ylabel(self.agg_name)
				a.legend(loc='best')
				variable_name = chosen_row['variable'][0]
				print('variable_name is ')
				print(variable_name)

				Intercept_value = round((chosen_row['param']['Intercept']),2)
				slope_name = list(chosen_row['param'])[1]
				slope_value = float(chosen_row['param'][slope_name])
				print("slope_value is :")
				print(slope_value)

				# Xuniques, X = np.unique(pattern_data_df[variable_name], return_inverse=True)
				print('pattern_data_df variable_name min value is:')
				print(pattern_data_df[variable_name].min())
				print('pattern_data_df variable_name max value is:')
				print(pattern_data_df[variable_name].max())

				var_min = pd.to_numeric(pattern_data_df[variable_name].min()).item()
				var_max = pd.to_numeric(pattern_data_df[variable_name].max()).item()
				print("var_min is: ")
				print(var_min)
				print("var_max is: ")
				print(var_max)
				X1 = np.linspace(var_min-2,var_max+2,100)
				dot_min = min(X1)
				dot_max = max(X1)
				print('X1 is:')
				print(X1)
				X = pattern_data_df[variable_name].astype('int64')
				print("X values:")
				print(X)
				Y = pattern_data_df[self.agg_name]
				print("Y values:")
				print(Y)
				y_vals = slope_value * X1 + Intercept_value
				print("y_vals are:")
				print(y_vals)
				a.plot(X1, y_vals, c='r',linewidth=2,label='Model')
				a.scatter(X, Y, s=20, c='b')
				a.legend(loc='best')
				a.set_xlim([min(var_min,dot_min)-1,max(var_max,dot_max)+1])
				a.set_xlabel(variable_name)
				a.set_ylabel(self.agg_name)

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

		
	def pop_up_explanation(self):

		win = Toplevel()
		win.geometry("%dx%d%+d%+d" % (1580, 700, 250, 125))
		win.wm_title("Explanation")

		win_frame = Frame(win)
		win_frame.pack(fill=BOTH,expand=True)
		win_frame.columnconfigure(0,weight=2)
		win_frame.columnconfigure(1,weight=3)
		win_frame.rowconfigure(0,weight=4)
		win_frame.rowconfigure(1,weight=1)

		b = ttk.Button(win_frame, text="Quit", command=win.destroy)
		b.grid(column=0,row=1,sticky='nsew')

		graph_frame = Frame(win_frame)
		graph_frame.grid(column=1,row=0,rowspan=2,sticky='nesw')

		f = Figure(figsize=(10,10),dpi=130)
		a = f.add_subplot(111)
		
		for n in self.exp_table.multiplerowlist:	
			relevent_pattern = self.exp_df.iloc[int(n)]['From_Pattern']
			rel_pattern_part = relevent_pattern.split(' : ')[0].split(' = ')[0].strip('[')
			rel_pattern_part_value = relevent_pattern.split(' : ')[0].split(' = ')[1].split(']')
			rel_pattern_pred = relevent_pattern.split(' : ')[1].split('>>')[0]
			rel_pattern_model = relevent_pattern.split(' : ')[1].split('>>')[1]
			rel_param = self.exp_df.iloc[int(n)]['relevent_param']
			rel_pattern_part_list = rel_pattern_part.split(',')
			rel_pattern_pred_list = rel_pattern_pred.split(',')
			rel_pattern_part_value_list = rel_pattern_part_value[0].split(',')
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
		" FROM pub WHERE " + "("+rel_pattern_part+") = ("+query_pattern_value+")"+\
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
				a.set_ylabel(self.agg_name)
				a.legend(loc='best')

				Xuniques, X = np.unique(exp_pattern_df[variable_name], return_inverse=True)
				print('Xuniques:')
				print(Xuniques)
				print('X values:')
				print(X)
				Y = exp_pattern_df['sum_pubcount']
				print('Y:')
				print(Y)
				a.scatter(X, Y, s=60, c='b',label=self.agg_name)

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

				a.scatter(X, Y, exp_pattern_df['sum_pubcount'],s=60, c='b',label=self.agg_name)
				a.scatter(X1,Y1,self.question_tuple['sum_pubcount'],s=150,marker='p',c='r',label='User Question')
				a.scatter(X2,Y2,exp_tuple_df['pred_value'],s=150,marker='X',c='g',label='Explanation')
				a.legend(loc='best')
				a.set_xlabel(x_name)
				a.set_ylabel(y_name)
				a.set_zlabel(self.agg_name)

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

		elif(rel_pattern_model=='linear'):

			if(len(rel_pattern_pred.split(','))==1):

				variable_name = rel_pattern_pred.split(',')[0]
				f = Figure(figsize=(5,5),dpi=130)
				a = f.add_subplot(111)
				a.set_title("Explanation Graph")
				a.set_xlabel('Variable')
				a.set_ylabel(self.agg_name)
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
				a.scatter(X, Y, s=60, c='b',label=self.agg_name)
				a.scatter(X2,Y2,s=150,marker='p',c='r',label='User Question')
				a.scatter(X3,exp_tuple_df['pred_value'],s=150,marker='X',c='g',label='Explanation')
				a.legend(loc='best')
				a.set_xlim([min(var_min,dot_min)-1,max(var_max,dot_max)+1])
				a.set_xlabel(variable_name)
				a.set_ylabel(self.agg_name)

			canvas = FigureCanvasTkAgg(f,graph_frame)
			canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
			canvas.draw()

		likelihood_words = []

		if(exp_tuple_score<=0):
			likelihood_words = ['unlikely','not similar','slighlty']
		elif(exp_tuple_score<=10):
			likelihood_words = ['plausible','somewhat similar','']
		else:
			likelihood_words = ['highly plausible','similar','extremly']

		ranking_clause = "  This explanation was ranked "+ likelihood_words[0] + " because the counterbalance\nis " + likelihood_words[1]+\
		" to the user question and it deviates"+likelihood_words[2]+"\nfrom the predicted outcome.\n"

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
		fixed_attr_list = str(self.chosen_local_pattern['partition']).split(',')
		fixed_value_list = str(self.chosen_local_pattern['partition_values']).split(',')

		for n in range(len(fixed_attr_list)):
			eq = (fixed_attr_list[n]+"="+fixed_value_list[n])
			fixed_pair_list.append(eq)
		if(len(fixed_pair_list)==1):
			fixed_pair = fixed_pair_list[0]
		else:
			fixed_pair = ",".join(fixed_pair_list)

		variable_pair_list=[]
		variable_attr_list = str(self.chosen_local_pattern['predictor']).split(',')

		for n in range(len(variable_attr_list)):
			eq = (str(variable_attr_list[n])+"="+str(self.question_tuple[variable_attr_list[n]].to_string(index=False)))
			variable_pair_list.append(eq)
		if(len(variable_pair_list)==1):
			variable_pair = variable_pair_list[0]
		else:
			variable_pair = ",".join(variable_pair_list)

		counter_dir = ''

		if(str(self.question['direction'])=='high'):
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


		comprehensive_exp = "\n\n  Explanation for why sum(pubcount) is " + self.question['direction']+"er than expected for:\n"+user_question_clause+\
		"\n  In general, "+str(rel_pattern_pred)+" "+predict+" sum(pubcount) for most "+str(rel_pattern_part)+"."+\
		"\nThis is also true for "+ fixed_pair+'.'\
		"\n  However, for "+variable_pair+",the value of sum(pubcount) is \n"+ self.question['direction']+"er than predicted."+\
		"\n  This may be explained through the "+counter_dir+"er than expected outcome \nfor "+ exp_clause+"."

		comprehensive_exp = comprehensive_exp.replace('name','author')
		comprehensive_exp = comprehensive_exp.replace('\'','')

		print('comprehensive_exp:')
		print(comprehensive_exp)

		pattern_description = Label(win_frame,text=ranking_clause+comprehensive_exp,font=('Times New Roman bold',18),borderwidth=5,relief=SOLID,justify=LEFT)
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




