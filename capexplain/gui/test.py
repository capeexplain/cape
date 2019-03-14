import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib.figure import Figure
from math import floor,ceil


test_df = pd.DataFrame({'name':['sigmod','sigmod','bbb','ccc'],'year':['2012','2014','2016','2018'],'sum':['5.2','3.7','10','20.3']})
question_df = pd.DataFrame({'name':['ccc'],'year':['2018'],'sum':['20.3']})


# test_df[['year','sum']] = test_df[['year','sum']].apply(lambda x : pd.to_numeric(x))
# question_df[['year','sum']] = question_df[['year','sum']].apply(lambda x : pd.to_numeric(x))

# print(test_df['year'])
# print(test_df['sum'])
# fig = plt.figure()
# ax = fig.gca(projection='3d') 
# var= 'name'

# test_df['coded_'+var] = test_df['name'].astype('category').cat.codes
# dict1 = dict(zip(test_df[var],test_df['coded_'+var]))
# dict1['dict_name'] = 'coded_'+var

# question_df['coded_'+var] = question_df[var].map(dict1)

# test_df

# print(test_df)
# pattern_only_df = pd.concat([test_df,question_df, question_df]).drop_duplicates(keep=False)

# ax.scatter(pattern_only_df['year'],pattern_only_df['coded_'+var],pattern_only_df['sum'],c='g',s=60,alpha=0.8,zorder=0,label="Pattern")
# ax.set_yticks(test_df['coded_'+var].values)
# ax.set_yticklabels(test_df[var])
# green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")

# ax.legend([green_proxy],['cars'])


# print(question_df)
# ax.scatter(question_df['year'],question_df['coded_'+var],question_df['sum'],c='red',s=200,alpha=1,zorder=10,label="Question")

# red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
# ax.legend([green_proxy],['cars'])
# plt.show()


# result = pd.merge(test_df,question_df,on=['name'])

# print(result)


# print(test_df.items())
# user_question_list=[]

# for k,v in question_df.items():
# 	user_question_list.append(str(k)+"="+str(v.to_string(index=False)))
# user_question_clause = ','.join(user_question_list)

# print(user_question_clause)


def encrypt(string, length):
    return '\n'.join(string[i:i+length] for i in range(0,len(string),length))

string = """
  Explanation for why sum(pubcount) is higher than expected for:author=Aaron B. Wagner,venue=corr,year=2010.  In general, year predicts sum(pubcount) for most author.This is also true for author=Aaron B. Wagner.
"""

str1 = encrypt(string,50)

print(str1)