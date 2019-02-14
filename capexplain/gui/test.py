import re

# from_which = re.compile('from (.*)',re.IGNORECASE)

# line = "select * from table_name"
# line1 ="SELECT * FROM table_name1"

# print(from_which.search(line1).group(1))



# str1 = "Pub"
# print(str1.lower())

# agg_function = re.compile('.*(sum|max|avg|min|count).*')

# agg_alias = re.compile('as\s+(\w+)',re.IGNORECASE) 

# text = "+\
# SELECT primary_type,year,count(*) AS count "+\
# "FROM crime"+\
# "GROUP BY primary_type,year;"

# print(agg_function.search(text).group(1))

# print(agg_alias.search(text).group(1))

list1 = ['year','venue']
list2 = ['year']

print(','.join([x for x in list1 if x not in list2]))