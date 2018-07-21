class CategorySimilarityMatrix(object):
    """ Similarity measure for categorical attributes defined manually
    """
    def compute_similarity(self, col, val1, val2, aggr_col):
        # print(col, val1, val2, self.sim_matrix[col][val1])
        if col not in self.sim_matrix:
            return -1
        else:
            if val1 not in self.sim_matrix[col] and val2 not in self.sim_matrix[col]:
                return -1
            return self.sim_matrix[col][val1][val2]

    def __init__(self, inf=''):
        if inf == '':
            return
        infile = open(inf, 'r')
        self.sim_matrix = {}
        col = ''
        while True:
            line = infile.readline()
            if not line:
                break
            temp_arr = line.split(',')
            if len(temp_arr) == 1:
                col = line.strip()
                self.sim_matrix[col] = {}
            else:
                sim_tuple = temp_arr
                if sim_tuple[0] not in self.sim_matrix[col]:
                    self.sim_matrix[col][sim_tuple[0]] = {sim_tuple[0]:1.0}
                if sim_tuple[1] not in self.sim_matrix[col]:
                    self.sim_matrix[col][sim_tuple[1]] = {sim_tuple[1]:1.0}
                self.sim_matrix[col][sim_tuple[0]][sim_tuple[1]] = float(sim_tuple[2])
                self.sim_matrix[col][sim_tuple[1]][sim_tuple[0]] = float(sim_tuple[2])


    def is_categorical(self, col):
        return col in self.sim_matrix