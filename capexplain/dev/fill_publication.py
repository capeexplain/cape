import pandas

DEFAULT_RESULT_PATH = './publication.csv'


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_integer(s):
    try:
        if s[-1] == 'L':
            int(s[0:-1])
        else:
            int(s)
        return True
    except ValueError:
        return False

def float_or_integer(s):
    if is_float(s):
        return float(s)
    elif is_integer(s):
        return int(s[0:-1]) if s[-1] == 'L' else int(s)
    else:
        return s

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
    #     # if it is a categorical attribute, first encode each one into integers, and then use one-hot encoding
    #     if df[column].dtype.kind == 'S' or df[column].dtype.kind == 'O':
    #         le[column] = preprocessing.LabelEncoder()
    #         le[column].fit(df[column])
    #         ohe[column] = preprocessing.OneHotEncoder()
    #         le_col = le[column].transform(df[column])
    #         le_col = le_col.reshape(-1, 1)
    #         ohe[column] = preprocessing.OneHotEncoder(sparse=False)
    #         ohe[column].fit(le_col)
    # df.insert(0, 'index', range(0, len(df))) 
    # data = {'df':df, 'le':le, 'ohe':ohe}
    return df

def main(args):
    query_result_file = DEFAULT_RESULT_PATH
    
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

    df = load_data(query_result_file)
    last_row = None
    outfile = open('publication_filled.csv', 'w')
    outfile.write(df.columns)
    outfile.write('\n')
    for row in df.iterrows():
        if last_row is not None:
            if row['name'] == last_row['name'] and row['pubkey'] == last_row['pubkey']:
                for y in range(last_row['year']+1, row['year'])
                    outfile.write('{},{},{},{}\n'.format(row['name'], row['pubkey'], str(y), 0))
        outfile.write(str(row) + '\n')
        last_row = row

    outfile.close()

if __name__ == '__main__':
    main(sys.argv[1:])