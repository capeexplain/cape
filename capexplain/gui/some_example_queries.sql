SELECT name,venue,year,sum(pubcount) as sum_pubcount
from pub_large_no_domain
group by name,venue,year