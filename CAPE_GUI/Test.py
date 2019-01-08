import re

agg_function = re.compile('.*(sum|max|avg|min|count).*')
group_by = re.compile("group by(.*)",re.IGNORECASE)

Q = """
select primary_type, location_description, year, count(*) as number 
from crime_10000
GrOup by primary_type, location_description, year
"""

query_list = Q.split('\n')
print(query_list)

agg = None
group_list = []
final_group_list = []
for line in query_list:
	if(agg_function.search(line) is not None):
		agg = agg_function.search(line).group(1)
	if(group_by.search(line) is not None):
		group_list = group_by.search(line).group(1).split(',')
for n in group_list:
	n = n.strip()
	final_group_list.append(n)

print("agg is "+str(agg))
print(group_list)
print(final_group_list)

