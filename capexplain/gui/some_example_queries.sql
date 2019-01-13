SELECT name,year,venue,sum(pubcount) as sum_count
from pub_large_no_domain
group by name,year,venue 

SELECT name,year,sum(pubcount) as sum_count
from pub_large_no_domain
group by name,year

select primary_type,description,location_description,year,count(*) as sumg
from crime_100000
group by primary_type,description,location_description,year



with user_query as ( select primary_type,description,location_description,year,count(*) as sumg from crime_100000 group by primary_type,description,location_description,year)
SELECT * FROM user_query uq where primary_type = 'ROBBERY'
INTERSECT
SELECT * FROM user_query uq where location_description = 'HOTEL/MOTEL'
