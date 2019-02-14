SELECT name,venue,year,sum(pubcount) AS sum_pubcount
FROM pub
GROUP BY name,venue,year;


SELECT primary_type,year,count(*) as count
FROM crime
GROUP BY primary_type,year;
