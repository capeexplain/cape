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

agg_function = re.compile('.*(sum|max|avg|min|count).*')
group_by = re.compile("group by(.*)",re.IGNORECASE)
float_num = re.compile('\d+\.\d+')

conn = psycopg2.connect(dbname="antiprov",user="antiprov",host="127.0.0.1",port="5436")

# "dbname = math564 user = chenjie password = lcj53242 \
#                       host = newbballserver.ctkmtyhjwqb1.us-east-2.rds.amazonaws.com"
cur = conn.cursor() # activate cursor


test_frame = pd.read_csv('/home/chenjie/Desktop/team_stats.csv')

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

		self.pattern = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=760)
		self.pattern.grid(column=2, row=0, columnspan=1, rowspan=5, sticky='nsew')

		self.explaination = ttk.Frame(self.main_frame, borderwidth=5, relief="sunken",width=760)
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
		self.query_frame.columnconfigure(3,weight=1)

		self.query_info = Label(self.query_frame,text="Query Input")
		self.query=''
		self.query_info.grid(column=0,row=0)
		self.query_entry = Text(self.query_frame, height=8, width=80)
		self.query_entry.grid(column=0,row=1)
		self.query_button = Button(self.query_frame,text="Run",command=self.run_query)
		self.query_button.grid(column=2,row=2)
		self.show_pattern_button = Button(self.query_frame,text="Show Pattern",command=self.show_pattern)
		self.show_pattern_button.grid(column=3,row=2)

#----------------------------Query result frame --------------------------------#
		self.query_result.columnconfigure(0,weight=15)
		self.query_result.columnconfigure(1,weight=1)
		self.query_result.rowconfigure(0,weight=1)

		self.show_results = Frame(self.query_result)
		self.show_results.grid(column=0,row=0,sticky='nsew')

		self.query_result_table = Table(self.show_results)
		self.query_result_table.show()



#-------------------------------pattern frame---------------------------------------#
		self.pattern.rowconfigure(0,weight=1)
		self.pattern.rowconfigure(1,weight=10)
		self.pattern.rowconfigure(2,weight=1)
		self.pattern.columnconfigure(0,weight=10)

		self.show_patterns = Frame(self.pattern)
		self.show_patterns.grid(column=0,row=1,sticky='nsew')

		self.pattern_label = Label(self.pattern,text="Patterns")
		self.pattern_label.grid(column=0,row=0)

		self.pattern_filter_button = Button(self.pattern,text='Filter Pattern',command=self.filter_pattern)
		self.pattern_filter_button.grid(column=0,row=2)

		self.pattern_table = Table(self.show_patterns)
		self.pattern_table.show()

		raw_pattern_query = "select CONCAT(fixed,',',variable) as set, * from dev.crime_partial_local;"

		self.raw_pattern_df = pd.read_sql(raw_pattern_query,conn)

		self.raw_pattern_df = self.raw_pattern_df.drop(['theta','dev_pos','dev_neg'],axis=1)

		self.raw_pattern_df['stats'] = self.raw_pattern_df['stats'].str.split(',',expand=True)[0]

		self.raw_pattern_df['stats'] = self.raw_pattern_df['stats'].str.strip('[')









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


	def show_pattern(self):

		query_list = self.query.split('\n')
		# n=0
		# for line in query_list:
		#   print(n)
		#   print(line)
		#   n+=1


		self.pattern_df = self.raw_pattern_df
		self.pattern_df['set'] = self.pattern_df['set'].apply(self.delete_parenthesis)

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

		self.pattern_df = self.pattern_df[self.pattern_df.set.apply(lambda x: x in query_group_str)]
		
		for index, row in self.pattern_df.iterrows():
			row_list = row['set'].split(',')
			for n in row_list:
				if n not in query_group_set:
					self.pattern_df = self.pattern_df.drop(index)
		
		self.output_pattern_df = self.pattern_df.drop(columns=['set'],axis=1)

		pattern_model = TableModel(dataframe=self.output_pattern_df)
		self.pattern_table.updateModel(pattern_model)
		self.pattern_table.redraw()

	def filter_pattern(self):

		# original_pattern_df = pd.DataFrame.copy(self.output_pattern_df)
		original_output_df = pd.DataFrame.copy(self.query_result_df)

		# print(self.output_pattern_df.head())

		print(self.pattern_table.multiplerowlist)
		# print(self.pattern_table.multiplecollist)

		pattern_df_lists = []

		for n in self.pattern_table.multiplerowlist:
			
			pattern_fixed = self.output_pattern_df.iloc[int(n)]['fixed']
			print(pattern_fixed)

			pattern_fixed_value = self.output_pattern_df.iloc[int(n)]['fixed_value']
			print(pattern_fixed_value)

			pattern_tuples = list(zip(pattern_fixed, pattern_fixed_value))

			df = pd.DataFrame(pattern_tuples, columns=['fixed','fixed_value'])
			# print(df)
			pattern_df_lists.append(df)
		
		
		for index,row in original_output_df.iterrows():
			tuple_match_pattern = False 
			last_also_match = True

			for pattern_df in pattern_df_lists:
				for index1,row1 in pattern_df.iterrows():
					if(row[row1['fixed']]==row1['fixed_value'] and last_also_match==True):
						tuple_match_pattern=True
						last_also_match=True
					else:
						tuple_match_pattern=False
						last_also_match=False

				if(tuple_match_pattern==True):
					last_also_match=True
					break
				else:
					last_also_match=True
					continue

			if(tuple_match_pattern==False):
				original_output_df = original_output_df.drop(index)

		model = TableModel(dataframe=original_output_df)
		self.query_result_table.updateModel(model)
		self.query_result_table.redraw()







def main():
	root = Tk()
	root.title('CAPE')
	ui = CAPE_UI(root)
	root.mainloop()     

if __name__ == '__main__':
	main()




