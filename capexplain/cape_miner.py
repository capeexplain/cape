"""
Cape system for explaining outliers to aggregation queries.

cape_mine: mine patterns


"""

import sys
import getopt
from inspect import currentframe, getframeinfo
import sqlalchemy as sa
from capexplain.pattern_miner.PatternMiner import PatternFinder, MinerConfig

# ********************************************************************************
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
        engine = None

    def getUrl(self):
        print(self.__dict__)
        print('' if self.password is None else self.password)
        breakpoint()
        url = 'postgresql://{}{}@{}:{:d}/{}'.format(self.user,
                                                  ('' if self.password is None else self.password),
                                                  self.host,
                                                  self.port,
                                                  self.db)
        return url

    def connect(self):
        try:
            engine = sa.create_engine(self.getUrl(),
                echo=False)
        except Exception as ex:
            print(type(ex) + "\n\n" +  ex.args)
            sys.exit(1)
        return engine

    def close(self):
        if engine!=None:
            engine.dispose()

# ********************************************************************************
class ConfigOpt:

    longopt=None
    shortopt=None
    hasarg=False
    description=None
    value=None
    
    def __init__(self, longopt, shortopt=None, desc=None, hasarg=False):
        self.longopt = longopt
        self.shortopt = shortopt
        self.hasarg = hasarg
        self.description = desc
        self.value = None

    def helpString(self):
        helpMessage=''
        if self.shortopt != None:
            helpMessage='-' + self.shortopt + ' ,'
        helpMessage+= '--' + self.longopt
        if self.hasarg == True:
            helpMessage+= " <arg>"
        if self.description != None:
            helpMessage = '{:30}'.format(helpMessage)
            helpMessage+= " - " + self.description
        return helpMessage
        
            
# ********************************************************************************
def printHelp():
    helpMessage = 'cape-miner: Mining patterns that hold for a relation (necessary preprocessing step for generating explanations.\n\n'
    for opt in OPTIONS:
        helpMessage += opt.helpString() + "\n"
    print(helpMessage)

def defaultDBConfig():
    return DBConnection()

OPTIONS = [ ConfigOpt(longopt='help', desc='show this help message'),
            ConfigOpt(longopt='host', shortopt='h', desc='database connection host IP address', hasarg=True),
            ConfigOpt(longopt='user', shortopt='u', desc='database connection user', hasarg=True),
            ConfigOpt(longopt='password', shortopt='p', desc='database connection password', hasarg=True),
            ConfigOpt(longopt='db', shortopt='d', desc='database names', hasarg=True),
            ConfigOpt(longopt='port', shortopt='P', desc='database connection port', hasarg=True),
            ConfigOpt(longopt='target-table', shortopt='t', desc='mine patterns for this table', hasarg=True),
            ConfigOpt(longopt='gof-const', shortopt=None, desc='goodness-of-fit threshold for constant regression', hasarg=True),
            ConfigOpt(longopt='gof-linear', shortopt=None, desc='goodness-of-fit threshold for linear regression', hasarg=True),
            ConfigOpt(longopt='confidence', shortopt=None, desc='global confidence threshold', hasarg=True),
            ConfigOpt(longopt='regpackage', shortopt='r', desc=('regression analysis package to use {}'.format(MinerConfig.STATS_MODELS)), hasarg=True),
            ConfigOpt(longopt='local-support', shortopt=None, desc='local support threshold', hasarg=True),
            ConfigOpt(longopt='global-support', shortopt=None, desc='global support thresh', hasarg=True),
            ConfigOpt(longopt='fd-optimizations', shortopt='f', desc='activate functional dependency detection and optimizations'),
            ConfigOpt(longopt='algorithm', shortopt='a', desc='algorithm to use for pattern mining {}'.format(MinerConfig.ALGORITHMS), hasarg=True),
]

def constructOptions():
    shortopts = ''
    longopts = []
    shortopt_map = {}
    longopt_map = {}
    cmdConfig = {}
    for opt in OPTIONS:
        cmdConfig[opt.longopt] = opt # mapping from configuration option names to configuration objects
        if opt.shortopt != None:
            shortopt_map['-' + opt.shortopt] = opt # map short option to configuration object
            shortopts+=opt.shortopt
            if opt.hasarg:
                shortopts+=':'
        longopt_map['--' + opt.longopt] = opt
        if opt.hasarg:
            longopts.append(opt.longopt + '=')
        else:
            longopts.append(opt.longopt)
    return shortopts, longopts, shortopt_map, longopt_map, cmdConfig

def parseOptions(argv):
    config=MinerConfig()
    dbconn=defaultDBConfig()
    shortopts, longopts,shortopt_map, longopt_map, cmdConfig = constructOptions()
    
    try:
        opts, args = getopt.getopt(argv, shortopts, longopts)
    except getopt.GetoptError as e:
        print("Exception ", type(e), "\n\n", e.args, "\n")
        printHelp()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in shortopt_map:
            cmdopt = shortopt_map[opt]
            if cmdopt.hasarg:
                cmdopt.value = arg
            else:
                cmdopt.value = True
        elif opt in longopt_map:
            cmdopt = longopt_map[opt]
            if cmdopt.hasarg:
                cmdopt.value = arg
            else:
                cmdopt.value = True
        else:
            print("invalid option {}".format(opt))
            sys.exit(2)
        
    for opt in cmdConfig:
        if cmdConfig[opt].value != None:
            if opt == 'host':
                dbconn.host = cmdConfig[opt].value
            elif opt == 'user':
                dbconn.user = cmdConfig[opt].value
            elif opt == 'password':
                dbconn.password = cmdConfig[opt].value
            elif opt == 'db':
                dbconn.database= cmdConfig[opt].value
            elif opt == 'port':
                dbconn.port = cmdConfig[opt].value
            elif opt == 'target-table':
                config.table = cmdConfig[opt].value
            elif opt == 'gof-const':
                config.theta_c = cmdConfig[opt].value
            elif opt == "gof-linear":
                config.theta_l = cmdConfig[opt].value
            elif opt == "confidence":
                config.lamb = cmdConfig[opt].value
            elif opt == "local-support":
                config.supp_l = cmdConfig[opt].value
            elif opt == "global-support":
                config.supp_g = cmdConfig[opt].value
            elif opt == "regpackage":
                config.reg_package = cmdConfig[opt].value
            elif opt == "fd-optimizations":
                config.fd_check = True
            elif opt == "algorithm":
                config.algorithm = cmdConfig[opt].value
            elif opt == "help":
                printHelp()
                sys.exit(2)
            else:
                print("unhandled config option <{}>".format(cmdConfig[opt].longopt))
    
    config.validateConfiguration()
    return config, dbconn

def main(argv=sys.argv[1:]):
    config, dbconn = parseOptions(argv)
    #config=['216.47.152.61','5432','postgres','antiprov','test']
    config.conn = dbconn.connect()
    p=PatternFinder(config)
    #fd=[(['A'],['B']),(['A','B'],['C'])]
    #p.addFd(fd)
    p.findPattern()
    conn.close()
        
if __name__=="__main__":
    main(sys.argv[1:])
