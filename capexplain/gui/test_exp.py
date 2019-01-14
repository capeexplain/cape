from capexplain.explain.explanation import ExplanationGenerator
from capexplain.explain.explanation import ExplConfig
from capexplain.explain.explanation import *

config=ExplConfig()
e = ExplanationGenerator(config, None)
# print(e.conn)
e.initialize()
elist = e.do_explain_online({'name': 'Jiawei Han', 'venue': 'kdd', 'year': 2007, 'sum_pubcount': 1, 'lambda': 0.2, 'direction': 'low'})

for e in elist:
	print(type(e))
	print(e.to_string())



