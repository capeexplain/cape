"""
Cape system for explaining outliers to aggregation queries.

cape_mine: mine patterns


"""

import sys
import getopt
import sqlalchemy as sa
from capexplain.pattern_miner.PatternMiner import PatternFinder, MinerConfig

def printHelp():
    print('cape-miner -q <query_result_file> -c <constraint_file> -u \
        <user_question_file> -o <outputfile> -e <epsilon> -a <aggregate_column>')

class DBConnection:
    """
    DBConnection wraps an sqlalchemy database connection.
    """
    host='127.0.0.1'
    user='postgres'
    port=5432
    db='postgres'
    password=None
    
    engine=None
    
    def __init__ (self,
                  host='127.0.0.1',
                  user='postgres',
                  port=5432,
                  db='postgres',
                  password=None):
        self.host=host
        self.user=user
        self.port=port
        self.db=db
        self.password=password

    def getUrl(self):
        url = 'postgresql://'+user+':'+password+'@'+host+':'+port+'/'+db
        return url

    def connect(self):
        try:
            engine = sa.create_engine(self.getUrl(),
                echo=False)
        except Exception as ex:
            print(ex)
            sys.exit(1)
        return engine

    def close(self):
        if engine!=None:
            engine.dispose()
        
def defaultDBConfig():
    return DBConnection()

SHORT_OPTIONS = "h:u:p:d:P:t:r:a:f:"
LONG_OPTIONS = ["help",
                "host",
                "user",
                "password",
                "db",
                "port",
                "target-table",
                "gof-const",
                "gof-linear",
                "confidence",
                "regpackage",
                "local-support",
                "global-support",
                "fd-optimizations",
                "algorithm"]

def parseOptions(argv):
    config=MinerConfig()
    dbconfig=defaultDBConfig()
    try:
        opts, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("--help"):
            printHelp()
            sys.exit(2)
        elif opt in ("-h", "--host"):
            dbconfig.host = arg
        elif opt in ("-u", "--user"):
            dbconfig.user = arg
        elif opt in ("-p", "--password"):
            dbconfig.password = arg
        elif opt in ("-d", "--db"):
            dbconfig.database= arg
        elif opt in ("-P", "--port"):
            dbconfig.port = arg
        elif opt in ("-t", "--target-table"):
            config.table = arg
        elif opt in ("--gof-const"):
            config.theta_c = arg
        elif opt in ("--gof-linear"):
            config.theta_l = arg
        elif opt in ("--confidence"):
            config.lamb = arg
        elif opt in ("--local-support"):
            config.supp_l = arg
        elif opt in ("--global-support"):
            config.supp_g = arg
        elif opt in ("r", "--regpackage"):
            config.reg_package = arg
        elif opt in ("f", "--fd-optimizations"):
            config.fd_check = arg
        elif opt in ("a", "--algorithm"):
            config.algorithm = arg
    config.validateConfiguration()
    return config, dbconfig

def main(argv=sys.argv[1:]):
    config, dbconn = parseOptions(argv)
    #config=['216.47.152.61','5432','postgres','antiprov','test']
    conn = dbconfig.connect()
    
    p=PatternFinder(conn,'publication')
    #fd=[(['A'],['B']),(['A','B'],['C'])]
    #p.addFd(fd)
    p.findPattern()
    conn.close()
        
if __name__=="__main__":
    main(sys.argv[1:])
