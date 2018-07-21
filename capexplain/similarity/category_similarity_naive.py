import math

class CategorySimilarityNaive(object):
    """ Similarity measure for categorical attributes defined manually
    """
    def compute_similarity(self, col, val1, val2, aggr_col):
        if col not in self.cate_cols:
            return -1
        if col in self.vector_dict:
            if val1 not in self.vector_dict[col] or val2 not in self.vector_dict[col]:
                return -1
            dist = math.sqrt(sum(map(
                lambda x:(x[0]-x[1])*(x[0]-x[1]), 
                zip(self.vector_dict[col][val1], self.vector_dict[col][val2]
            ))))
            return 1.0 / (1.0+dist)
        if val1 == val2:
            return 1.0
        else:
            return 0.0

    def __init__(self, cur, table_name, embedding_table_list):
        type_query = '''SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{}';'''.format(table_name)
        cur.execute(type_query)
        res = cur.fetchall()
        self.cate_cols = {}
        self.vector_dict = {}
        for (col, dt) in res:
            if dt == 'boolean' or dt.find('character') != -1:
                self.cate_cols[col] = True
        for (col, embedding_table_name) in embedding_table_list:
            self.vector_dict[col] = {}
            read_query = '''SELECT *  FROM {} ;'''.format(embedding_table_name)
            cur.execute(read_query)
            res = cur.fetchall()
            for (x, lat, log) in res:
                self.vector_dict[col][x] = (lat, log)
        

    def is_categorical(self, col):
        return col in self.cate_cols