SELECT name,venue,year,sum(pubcount) AS sum_pubcount
FROM pub
GROUP BY name,venue,year