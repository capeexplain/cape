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

agg_function = re.compile('.*(sum|max|avg|min|count).*')
group_by = re.compile("group by(.*)",re.IGNORECASE)
float_num = re.compile('\d+\.\d+')

conn = psycopg2.connect(dbname="antiprov",user="antiprov",host="127.0.0.1",port="5436")

cur = conn.cursor() # activate cursor



class CAPE_UI:

	def __init__(self,parent):

#----------------------------main frame----------------------------------------#        
		self.parent=parent
		self.main_frame_style=ttk.Style()
		self.main_frame_style.configure('Main_Frame', background='#334353')
		self.main_frame=ttk.Frame(self.parent,padding=(3,3,12,12),width=2000, height=100)

		self.main_frame.columnconfigure(0, weight=1)
		self.main_frame.columnconfigure(1, weight=1)
		self.main_frame.columnconfigure(2, weight=1)

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

		self.local_pattern = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=760)
		self.local_pattern.grid(column=2, row=0, columnspan=1, rowspan=5, sticky='nsew')

		self.explaination = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=760)
		self.explaination.grid(column=2, row=5, columnspan=1, rowspan=5, sticky='nsew')


#---------------------------table frame-----------------------------------------#
		self.table_view = ttk.Treeview(self.table_frame,height=46)
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
		self.query_frame.columnconfigure(0,weight=8)
		self.query_frame.columnconfigure(1,weight=1)

		self.query_info = Label(self.query_frame,text="Query Input",font=('Times New Roman bold',12),borderwidth=5,relief=RIDGE)
		self.query=''
		self.query_info.grid(column=0,row=0,sticky='nsew')
		self.query_entry = Text(self.query_frame, height=8, width=80)
		self.query_entry.grid(column=0,row=1,rowspan=4)
		self.query_button = Button(self.query_frame,text="Run Query",font=('Times New Roman bold',12),command=self.run_query)
		self.query_button.grid(column=1,row=2)
		self.show_global_pattern_button = Button(self.query_frame,text="Show Global Pattern",font=('Times New Roman bold',12),command=self.show_global_pattern)
		self.show_global_pattern_button.grid(column=1,row=3)
		self.show_local_pattern_button = Button(self.query_frame,text="Show Local Pattern",font=('Times New Roman bold',12),command=self.show_local_pattern)
		self.show_local_pattern_button.grid(column=1,row=4)
		

#----------------------------Query result frame --------------------------------#
		self.query_result.columnconfigure(0,weight=8)
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


		self.high_button = Button(self.high_low_frame,text='High',font=('Times New Roman bold',12))
		self.high_button.grid(column=0,row=1)

		self.low_button = Button(self.high_low_frame,text='Low',font=('Times New Roman bold',12))
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
		self.show_global.columnconfigure(0,weight=10)

		self.global_pattern_label = Label(self.show_global,text="Global Patterns")
		self.global_pattern_label.grid(column=0,row=0,sticky='nsew')

		self.show_global_patterns = Frame(self.show_global)
		self.show_global_patterns.grid(column=0,row=1,sticky='nsew')

		self.global_pattern_filter_button = Button(self.high_low_frame,text='Filter \nLocal\n Pattern',font=('Times New Roman bold',12),command=self.use_global_filter_local)
		self.global_pattern_filter_button.grid(column=0,row=6)

		self.global_pattern_table = Table(self.show_global_patterns)
		self.global_pattern_table.show()

		raw_global_pattern_query = "select CONCAT(fixed,',',variable) as set,* from dev.crime_clean_100000_global;"

		self.raw_global_pattern_df = pd.read_sql(raw_global_pattern_query,conn)

		self.raw_global_pattern_df = self.raw_global_pattern_df.drop(['theta','dev_pos','dev_neg'],axis=1)




#------------------------------- local pattern frame---------------------------------------#
		self.local_pattern.rowconfigure(0,weight=1)
		self.local_pattern.rowconfigure(1,weight=20)
		self.local_pattern.rowconfigure(2,weight=1)
		self.local_pattern.columnconfigure(0,weight=10)

		self.local_show_patterns = Frame(self.local_pattern)
		self.local_show_patterns.grid(column=0,row=1,sticky='nsew')

		self.local_pattern_label = Label(self.local_pattern,text="Local Patterns",font=('Times New Roman bold',12),borderwidth=5,relief=RIDGE)
		self.local_pattern_label.grid(column=0,row=0,sticky='nsew')

		self.local_pattern_filter_button = Button(self.local_pattern,text='Filter Output',font=('Times New Roman bold',12),command=self.use_local_filter_output)
		self.local_pattern_filter_button.grid(column=0,row=2)

		self.local_pattern_table_frame = Frame(self.local_pattern)
		self.local_pattern_table_frame.grid(row=1,column=0,sticky='nsew')
		self.local_pattern_table = Table(self.local_pattern_table_frame)
		self.local_pattern_table.show()

		raw_local_pattern_query = "select CONCAT(fixed,',',variable) as set, * from dev.crime_clean_100000_local;"

		self.raw_local_pattern_df = pd.read_sql(raw_local_pattern_query,conn)

		self.raw_local_pattern_df = self.raw_local_pattern_df.drop(['theta','dev_pos','dev_neg'],axis=1)

		self.raw_local_pattern_df['stats'] = self.raw_local_pattern_df['stats'].str.split(',',expand=True)[0]

		self.raw_local_pattern_df['stats'] = self.raw_local_pattern_df['stats'].str.strip('[')



#---------------------------------explaination frame-----------------------------#

		



#----------------------------------Functions----------------------------------------#
	
	def delete_parenthesis(self,colstr):

		string1=str(colstr.replace("{","").replace("}",""))
		list1=string1.split(',')
		list1.sort()

		return  (','.join(list1))

	def run_query(self):
		
		self.query_result_table = Table(self.show_results)
		self.query_result_table.show()
		self.query = self.query_entry.get("1.0",END)
		print(self.query)
		self.query_result_df = pd.read_sql(self.query,conn)
		model = TableModel(dataframe=self.query_result_df)
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

		self.global_pattern_df = self.global_pattern_df[self.global_pattern_df.apply(lambda row: all(i in query_group_set for i in row.set.split(',')),axis=1)]

		# for index, row in self.global_pattern_df.iterrows():
		# 	row_list = row['set'].split(',')
		# 	for n in row_list:
		# 		if n not in query_group_set:
		# 			self.global_pattern_df = self.global_pattern_df.drop(index)
		
		self.global_output_pattern_df = self.global_pattern_df.drop(columns=['set'],axis=1)

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

		self.local_pattern_df = self.local_pattern_df[self.local_pattern_df.apply(lambda row: all(i in query_group_set for i in row.set.split(',')),axis=1)]
		
		self.local_output_pattern_df = self.local_pattern_df.drop(columns=['set'],axis=1)

		pattern_model = TableModel(dataframe=self.local_output_pattern_df)
		self.local_pattern_table.updateModel(pattern_model)
		self.local_pattern_table.redraw()

	def use_global_filter_local(self):

		# original_output_df = pd.DataFrame.copy(self.query_result_df)

		original_local_df = pd.DataFrame.copy(self.raw_local_pattern_df)

		print(self.global_pattern_table.multiplerowlist)

		pattern_df_lists = []

		for n in self.global_pattern_table.multiplerowlist:
			
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

			pattern_tuples = [[global_pattern_fixed,global_pattern_variable]]

			print(pattern_tuples)

			df = pd.DataFrame(pattern_tuples, columns=['fixed','variable'])
			# print(df)
			pattern_df_lists.append(df)

		filtered_df = pd.DataFrame(columns=['fixed','variable'])
		
		for pattern_df in pattern_df_lists:
			print(pattern_df)
			g_fixed=pattern_df['fixed'].to_string(index=False)
			print(g_fixed)
			g_variable=pattern_df['variable'].to_string(index=False)
			print(g_variable)
			Q1 ="SELECT * FROM dev.crime_clean_100000_local WHERE array_to_string(fixed, ',')=\'"+g_fixed+"\'AND array_to_string(variable, ',')=\'"+g_variable+'\';'
			l_result = pd.read_sql(Q1, conn)
			filtered_df = filtered_df.append(l_result,ignore_index=True)

		self.local_output_pattern_df = filtered_df

		model = TableModel(dataframe=filtered_df)
		self.local_pattern_table.updateModel(model)
		self.local_pattern_table.redraw()


	def use_local_filter_output(self):

		original_output_df = pd.DataFrame.copy(self.query_result_df)

		print(self.local_pattern_table.multiplerowlist)
		# print(self.pattern_table.multiplecollist)

		pattern_df_lists = []
		for n in self.local_pattern_table.multiplerowlist:
			
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
				q = "SELECT * FROM user_query uq where "+fixed_col_name+"=\'"+fixed_col_value+"\'"
				query_list.append(q)
			querybody = '\nINTERSECT\n'.join(query_list)
			full_query = user_query_view+querybody
			print("FULL QUERY IS:")
			print(full_query)
			one_df = pd.read_sql(full_query,conn)
			print('one df here!!!!!!')
			print(one_df)
			filtered_result_df = filtered_result_df.append(one_df,ignore_index=True)
		print("filtered_df here!!!!!!!!!!!!!!!!!!!!!!!")
		print(filtered_result_df)

		model = TableModel(dataframe=filtered_result_df)
		self.query_result_table.updateModel(model)
		self.query_result_table.redraw()

def main():
	root = Tk()
	root.title('CAPE')
	ui = CAPE_UI(root)
	root.mainloop()     

if __name__ == '__main__':
	main()




