import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib.figure import Figure
from math import floor,ceil


# test_df = pd.DataFrame({'name':['sigmod','sigmod','bbb','ccc'],'year':['2012','2014','2016','2018'],'sum':['5.2','3.7','10','20.3']})

# test_df[['year','sum']] = test_df[['year','sum']].apply(lambda x : pd.to_numeric(x))
# print(test_df['year'])
# print(test_df['sum'])
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# var= 'name'
# test_df['coded_'+var] = test_df['name'].astype('category').cat.codes
# print(test_df)
# ax.scatter(test_df['year'],test_df['coded_'+var],test_df['sum'],c='g',label='sum',s=60)
# ax.set_yticks(test_df['coded_'+var].values)
# ax.set_yticklabels(test_df[var])
# question_df = test_df.loc[test_df['name']=='bbb']
# print(question_df)
# ax.scatter(question_df['year'],question_df['coded_'+var],question_df['sum'],c='black',label='explanation',s=200,marker='P')

# plt.show()
test_df = pd.DataFrame({'name':['sigmod','sigmod','bbb','ccc'],'year':[2013,2014,2016,2018],'pubcount':[5.1,3.2,10.4,4.1]})

print(list(range(floor(test_df['pubcount'].values.min()),ceil(test_df['pubcount'].values.max()))))

# list_1 = list(range(0,10))
# print(list_1)