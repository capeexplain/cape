SELECT name,venue,year,sum(pubcount) AS sum_pubcount
FROM pub_large_no_domain
GROUP BY name,venue,year