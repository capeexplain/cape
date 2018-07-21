#!/usr/bin/python
# -*- coding:utf-8 -*- 

import sys, getopt
import pandas
import csv
#import statsmodels.formula.api as smf
from sklearn import preprocessing
import math
import time
from heapq import *

import operator

sys.path.append('./')
sys.path.append('../')
from similarity_calculation.category_similarity_matrix import *
from similarity_calculation.category_network_embedding import *
from utils import *
from constraint_definition.LocalRegressionConstraint import *


DEFAULT_RESULT_PATH = './input/query_res.csv'
DEFAULT_QUESTION_PATH = './input/user_question.csv'
DEFAULT_CONSTRAINT_PATH = './input/CONSTRAINTS'
EXAMPLE_NETWORK_EMBEDDING_PATH = './input/NETWORK_EMBEDDING'
EXAMPLE_SIMILARITY_MATRIX_PATH = './input/SIMILARITY_DEFINITION'
DEFAULT_AGGREGATE_COLUMN = 'count'
DEFAULT_CONSTRAINT_EPSILON = 0.05
TOP_K = 5

def build_local_regression_constraint(data, column_index, t, con, epsilon, agg_col, regression_package):
    """Build local regression constraint from Q(R), t, and global regression constraint

    Args:
        data: result of Q(R)
        column_index: index for values in each column
        t: target tuple in Q(R)
        con: con[0] is the list of fixed attributes in Q(R), con[1] is the list of variable attributes in Q(R)
        epsilon: threshold for local regression constraint
        regression_package: which package is used to compute regression 
    Returns:
        A LocalRegressionConstraint object whose model is trained on \pi_{con[1]}(Q_{t[con[0]]}(R))
    """
    tF = get_F_value(con[0], t)
    local_con = LocalRegressionConstraint(con[0], tF, con[1], agg_col, epsilon)
    train_data = {agg_col: []}
    for v in con[1]:
        train_data[v] = []
    # for index, row in data['df'].iterrows():
    #     if get_F_value(con[0], row) == tF:
    #         for v in con[1]:
    #             train_data[v].append(row[v])
    #         train_data[agg_col].append(row[agg_col])

    
    for idx in column_index[con[0][0]][tF[0]]:
        row = data['df'].loc[data['df']['index'] == idx]
        row = row.to_dict('records')[0]
        #print row
        if get_F_value(con[0], row) == tF:
            for v in con[1]:
                train_data[v].append(row[v])
            train_data[agg_col].append(row[agg_col])
    if regression_package == 'scikit-learn':
        train_x = {}
        for v in con[1]:
            if v in data['le']:
                train_data[v] = data['le'][v].transform(train_data[v])
                train_data[v] = data['ohe'][v].transform(train_data[v].reshape(-1, 1))
                #print data['ohe'][v].transform(train_data[v].reshape(-1, 1))
                train_x[v] = train_data[v]
            else:
                if v != agg_col:
                    train_x[v] = np.array(train_data[v]).reshape(-1, 1)
        train_y = np.array(train_data[agg_col]).reshape(-1, 1)
        train_x = np.concatenate(list(train_x.values()), axis=-1)
        local_con.train_sklearn(train_x, train_y)
    else:
        #train_data = pandas.DataFrame(train_data)
        formula = agg_col + ' ~ ' + ' + '.join(con[1])
        print 
        local_con.train(train_data, formula)
    return local_con

def validate_local_regression_constraint(data, local_con, t, dir, agg_col, regression_package):
    """Check the validicity of the user question under a local regression constraint

    Args:
        data: data['df'] is the data frame storing Q(R)
            data['le'] is the label encoder, data['ohe'] is the one-hot encoder
        local_con: a LocalRegressionConstraint object
        t: target tuple in Q(R)
        dir: whether user thinks t[agg(B)] is high or low
        agg_col: the column of aggregated value
        regression_package: which package is used to compute regression 
    Returns:
        the actual direction that t[agg(B)] compares to its expected value, and the expected value from local_con
    """
    test_tuple = {}
    for v in local_con.var_attr:
        test_tuple[v] = [t[v]]
    if regression_package == 'scikit-learn':
        for v in local_con.var_attr:
            if v in data['le']:
                test_tuple[v] = data['le'][v].transform(test_tuple[v])
                test_tuple[v] = data['ohe'][v].transform(test_tuple[v].reshape(-1, 1))
            else:
                test_tuple[v] = np.array(test_tuple[v]).reshape(-1, 1)
        
        test_tuple = np.concatenate(list(test_tuple.values()), axis=-1)
        predictY = local_con.predict_sklearn(test_tuple)
    else:
        predictY = local_con.predict(pandas.DataFrame(test_tuple))

    if t[agg_col] < (1-local_con.epsilon) * predictY[0]:
        return -dir, predictY[0]
    elif t[agg_col] > (1+local_con.epsilon) * predictY[0]:
        return dir, predictY[0]
    else:
        return 0, predictY[0]
        
def tuple_similarity(t1, t2, var_attr, cat_sim, num_dis_norm, agg_col):
    """Compute the similarity between two tuples t1 and t2 on their attributes var_attr

    Args:
        t1, t2: two tuples
        var_attr: variable attributes
        cat_sim: the similarity measure for categorical attributes
        num_dis_norm: normalization terms for numerical attributes
        agg_col: the column of aggregated value
    Returns:
        the Gower similarity between t1 and t2
    """
    sim = 0.0
    cnt = 0
    for col in var_attr:
        if t1[col] is None or t2[col] is None:
            continue
        if cat_sim.is_categorical(col):
            s = cat_sim.compute_similarity(col, t1[col], t2[col], agg_col)
            sim += s
        else:
            if col != agg_col and col != 'index':
                temp = abs(t1[col] - t2[col]) / num_dis_norm[col]['range']
                sim += 1-temp
        cnt += 1
    return sim / cnt

def find_best_user_questions(data, cons, cat_sim, num_dis_norm, cons_epsilon, agg_col, regression_package):

    """Find explanations for user questions

    Args:
        data: data['df'] is the data frame storing Q(R)
            data['le'] is the label encoder, data['ohe'] is the one-hot encoder
        cons: list of fixed attributes and variable attributes of global constraints
        cat_sim: the similarity measure for categorical attributes
        num_dis_norm: normalization terms for numerical attributes
        cons_epsilon: threshold for local regression constraints
        agg_col: the column of aggregated value
        regression_package: which package is used to compute regression 
    Returns:
        the top-k list of explanations for each user question
    """

    index_building_time = 0
    constraint_building_time = 0
    question_validating_time = 0
    score_computing_time = 0
    result_merging_time = 0

    start = time.clock()
    column_index = dict()
    for column in data['df']:
        column_index[column] = dict()
    for index, row in data['df'].iterrows():
        for column in data['df']:
            val = row[column]
            if not val in column_index[column]:
                column_index[column][val] = []
            column_index[column][val].append(index)
    end = time.clock()
    index_building_time += end - start

    psi = []
    # local_cons = []
    # start = time.clock()
    # for i in range(len(cons)):
    #     local_cons.append(build_local_regression_constraint(data, column_index, t, cons[i], cons_epsilon, agg_col, regression_package))
    #     local_cons[i].print_fit_summary()
    # end = time.clock()
    # constraint_building_time += end - start

    explanation_type = 0
    max_support = []
    candidates = []
    for i in range(0, len(cons)):
        psi.append(0)
        max_support.append([])
        f_indexes = dict()
        print(cons[i])

        for index, row in data['df'].iterrows():
            t = get_F_value(cons[i][0], row)
            if ','.join(t) in f_indexes:
                continue
                
            con_index = None
            for j in range(len(cons[i][0])):
                idx_j = column_index[cons[i][0][j]][t[j]]
                # print(idx_j)
                # print(data['df']['index'].isin(idx_j))
                if con_index is None:
                    con_index = pandas.Index(idx_j)
                else:
                    con_index = con_index.intersection(pandas.Index(idx_j))
                # print(con_index)


            selected_rows = data['df'].loc[data['df']['index'].isin(con_index)]
            des = selected_rows['count'].describe()
            if des.loc[['count']].values[0] > 7:
                if des.loc[['mean']].values[0] > 1.49:
                    print(t, des)
                    candidates.append([selected_rows, des.loc[['mean']].values[0], 
                        des.loc[['count']].values[0], des.loc[['std']].values[0], 
                        des.loc[['75%']].values[0], des.loc[['25%']].values[0]])
        
            f_indexes[','.join(t)] = selected_rows
            data['df'].drop(con_index)

            # break
            
            # avg = sum()
            
            # rows = data['df'].loc[]
            # print(rows)
        print("i ", i)
        break
                
    return sorted(candidates, key=lambda x: x[4]-x[5])

def load_data(qr_file=DEFAULT_RESULT_PATH):
    ''' 
        load query result
    '''
    df = pandas.read_csv(open(qr_file, 'r'), header=0, quotechar="'")
    le = {}
    ohe = {}
    for column in df:
        df[column] = df[column].apply(lambda x: x.replace('\'', '').strip())
        df[column] = df[column].apply(lambda x: float_or_integer(x))
        # if it is a categorical attribute, first encode each one into integers, and then use one-hot encoding
        if df[column].dtype.kind == 'S' or df[column].dtype.kind == 'O':
            le[column] = preprocessing.LabelEncoder()
            le[column].fit(df[column])
            ohe[column] = preprocessing.OneHotEncoder()
            le_col = le[column].transform(df[column])
            le_col = le_col.reshape(-1, 1)
            ohe[column] = preprocessing.OneHotEncoder(sparse=False)
            ohe[column].fit(le_col)
    df.insert(0, 'index', range(0, len(df))) 
    data = {'df':df, 'le':le, 'ohe':ohe}
    return data

def load_user_question(uq_path=DEFAULT_QUESTION_PATH):
    '''
        load user questions
    '''
    uq = []
    with open(uq_path, 'rt') as uqfile:
        reader = csv.DictReader(uqfile, quotechar='\'')
        headers = reader.fieldnames
        #temp_data = csv.reader(uqfile, delimiter=',', quotechar='\'')
        #for row in temp_data:
        for row in reader:
            row_data = {}
            for k, v in enumerate(headers):
                print(k, v)
                if v != 'direction':
                    if is_float(row[v]):
                        row_data[v] = float(row[v])
                    elif is_integer(row[v]):
                        row_data[v] = float(long(row[v]))
                    else:
                        row_data[v] = row[v]
            if row['direction'][0] == 'h':
                dir = 1
            else:
                dir = -1
            uq.append({'target_tuple': row_data, 'dir':dir})
    return uq

def load_constraints(cons_path=DEFAULT_CONSTRAINT_PATH):
    '''
        load pre-defined constraints(currently only fixed attributes and variable attributes)
    '''
    inf = open(cons_path, 'r')
    cons = []
    while True:
        line = inf.readline()
        if not line:
            break
        cons.append([[],[]])
        cons[-1][0] = line.strip(' \r\n').split(',')
        line = inf.readline()
        cons[-1][1] = line.strip(' \r\n').split(',')
    inf.close()
    return cons

        
def main(argv=[]):
    query_result_file = DEFAULT_RESULT_PATH
    constraint_file = DEFAULT_CONSTRAINT_PATH
    user_question_file = DEFAULT_QUESTION_PATH
    outputfile = ''
    constraint_epsilon = DEFAULT_CONSTRAINT_EPSILON
    aggregate_column = DEFAULT_AGGREGATE_COLUMN
    try:
        opts, args = getopt.getopt(argv,"hq:c:u:o:e:a",["qfile=","cfile=","ufile=","ofile=","epsilon=","aggregate_column="])
    except getopt.GetoptError:
        print('explanation.py -q <query_result_file> -c <constraint_file> -u \
        <user_question_file> -o <outputfile> -e <epsilon> -a <aggregate_column>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('explanation.py -q <query_result_file> -c <constraint_file> -u \
            <user_question_file> -o <outputfile> -e <epsilon> -a <aggregate_column>')
            sys.exit(2)    
        elif opt in ("-q", "--qfile"):
            query_result_file = arg
        elif opt in ("-c", "--cfile"):
            constraint_file = arg
        elif opt in ("-u", "--ufile"):
            user_question_file = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-e", "--epsilon"):
            constraint_epsilon = float(arg)
        elif opt in ("-a", "--aggcolumn"):
            aggregate_column = arg

    start = time.clock()
    data = load_data(query_result_file)
    constraints = load_constraints(DEFAULT_CONSTRAINT_PATH)
    Q = load_user_question(user_question_file)
    category_similarity = CategorySimilarityMatrix(EXAMPLE_SIMILARITY_MATRIX_PATH)
    #category_similarity = CategoryNetworkEmbedding(EXAMPLE_NETWORK_EMBEDDING_PATH, data['df'])
    num_dis_norm = normalize_numerical_distance(data['df'])
    end = time.clock()
    print('Loading time: ' + str(end-start) + 'seconds')
    
    start = time.clock()
    regression_package = 'scikit-learn'
    #regression_package = 'statsmodels'
    # explanations_list = find_explanation_regression_based(data, Q, constraints, category_similarity, 
    #                                                       num_dis_norm, constraint_epsilon, 
    #                                                       aggregate_column, regression_package)
    uq_list = find_best_user_questions(data, constraints, category_similarity, 
                                                          num_dis_norm, constraint_epsilon, 
                                                          aggregate_column, regression_package)
    end = time.clock()
    print('Total querying time: ' + str(end-start) + 'seconds')

    # ofile = sys.stdout
    ofile = open('./candidate_authors.txt', 'w')
    for i in range(len(uq_list)):
        #print(uq_list[i])
        ofile.write(str(uq_list))
    ofile.close()
 

if __name__ == "__main__":
    main(sys.argv[1:])


