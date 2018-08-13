"""
Commandline tool capexplain for the Cape system which explains outliers to aggregation queries.

Installation through setuptools will create a script capexplain. Run "capexplain help" to see usage.
"""

import sys
from enum import Enum, unique
import getopt
import logging
from inspect import currentframe, getframeinfo
from capexplain.pattern_miner.PatternMiner import PatternFinder, MinerConfig
from capexplain.database.dbaccess import DBConnection
from capexplain.cl.cfgoption import ConfigOpt, OptionType
from capexplain.cl.command import CmdTypes, Command, CmdOptions
from capexplain.explain.explanation import ExplanationGenerator, ExplConfig
import colorful

# ********************************************************************************
# format for logging
LOGFORMAT='{c.white_on_black}%(levelname)s{c.reset} {c.red}%(asctime)s{c.reset} {c.blue}[%(filename)s:%(funcName)s:%(lineno)d]{c.reset} %(message)s'.format(c=colorful)


# ********************************************************************************
def mineCommand(c,log):
    """
    Command action for mine for patterns (options are parsed commandline options).
    """
    # create configuration based on options
    config=MinerConfig()
    dbconn=DBConnection()

    log.debug("executing mine command")

    o = c.options
    for opt in o.cmdConfig:
        option = o.cmdConfig[opt]
        if option.value != None:
            key =  opt if (option.cfgFieldName is None) else option.cfgFieldName
            val = option.value
            log.debug("option: {}:{}".format(key,val))
            if key in dbconn.getValidKeys():
                dbconn[key] = val
            elif key in config.getValidKeys():
                config[key] = val
            else:
                log.warning("unhandled config option <{}>".format(option.longopt))
    
    config.validateConfiguration()
    config.conn = dbconn.connect()
    
    log.debug("connected to database")
    p=PatternFinder(config)
    log.debug("created pattern miner object")
    p.findPattern()
    log.debug("done finding patterns")
    config.conn.close()
    log.debug("closed database connection ... DONE")

# ********************************************************************************
def helpCommand(argv):
    """
    Command for showing help message.
    """
    if len(argv) == 0:
        printHelp()
    else:
        cmdString = argv[0]
        command = next((c for c in COMMANDS if c.cmdstr == cmdString), None)
        if command is None:
            print("unknown command: {}\n\n".format(cmdString))
            printHelp()
        else:
            printHelp(command)

# ********************************************************************************
def statsCommand(command,log):
    """
    Command for printing stats and patterns of previous executions of miner or explainer.
    """
    # create configuration based on options
    config=DictLike()
    dbconn=DBConnection()

    log.debug("executing mine command")

    o = c.options
    for opt in o.cmdConfig:
        option = o.cmdConfig[opt]
        if option.value != None:
            key =  opt if (option.cfgFieldName is None) else option.cfgFieldName
            val = option.value
            log.debug("option: {}:{}".format(key,val))
            if key in dbconn.getValidKeys():
                dbconn[key] = val
            elif key in config.getValidKeys():
                config[key] = val
            else:
                log.warning("unhandled config option <{}>".format(option.longopt))
    
    config.validateConfiguration()
    config.conn = dbconn.connect()
    log.debug("connected to database")
    config.conn.close()
    log.debug("closed database connection ... DONE")

# ********************************************************************************
def explainCommand(command,log):
    """
    Command for explaining an outlier.
    """
    # create configuration based on options
    config=ExplConfig()

    # setup configuration
    command.options.setupConfig(config)

    # do explaining
    log.debug("executing explain command")
    e = ExplanationGenerator(config)
    log.debug("created ExplanationGenerator")
    e.doExplain()
    log.debug("explanation generation finished")
    

# ********************************************************************************
# options for difference commands using ConfigOpt
COMMON_OPTIONS =  [ ConfigOpt(longopt='log', shortopt='l', desc='select log level {DEBUG,INFO,WARNING,ERROR}', hasarg=True, value="ERROR"),
                    ConfigOpt(longopt='help', desc='show this help message'),
]

MINE_OPTIONS = COMMON_OPTIONS + [ ConfigOpt(longopt='host', shortopt='h', desc='database connection host IP address', hasarg=True),
            ConfigOpt(longopt='user', shortopt='u', desc='database connection user', hasarg=True),
            ConfigOpt(longopt='password', shortopt='p', desc='database connection password', hasarg=True),
            ConfigOpt(longopt='db', shortopt='d', desc='database name', hasarg=True),
            ConfigOpt(longopt='port', shortopt='P', desc='database connection port', hasarg=True,otype=OptionType.Int),
            ConfigOpt(longopt='target-table', shortopt='t', desc='mine patterns for this table', hasarg=True, cfgFieldName='table'),
            ConfigOpt(longopt='gof-const', shortopt=None, desc='goodness-of-fit threshold for constant regression', hasarg=True, otype=OptionType.Float, cfgFieldName='theta_c'),
            ConfigOpt(longopt='gof-linear', shortopt=None, desc='goodness-of-fit threshold for linear regression', hasarg=True, otype=OptionType.Float, cfgFieldName='theta_l'),
            ConfigOpt(longopt='confidence', shortopt=None, desc='global confidence threshold', hasarg=True, otype=OptionType.Float, cfgFieldName='lamb'),
            ConfigOpt(longopt='regpackage', shortopt='r', desc=('regression analysis package to use {}'.format(MinerConfig.STATS_MODELS)), hasarg=True,cfgFieldName='reg_package'),
            ConfigOpt(longopt='local-support', shortopt=None, desc='local support threshold', hasarg=True, otype=OptionType.Int,cfgFieldName='supp_l'),
            ConfigOpt(longopt='global-support', shortopt=None, desc='global support thresh', hasarg=True, otype=OptionType.Int,cfgFieldName='supp_g'),
            ConfigOpt(longopt='fd-optimizations', shortopt='f', desc='activate functional dependency detection and optimizations',cfgFieldName='fd_check'),
            ConfigOpt(longopt='algorithm', shortopt='a', desc='algorithm to use for pattern mining {}'.format(MinerConfig.ALGORITHMS), hasarg=True),
            ConfigOpt(longopt='show-progress', shortopt=None, desc='show progress meters', otype=OptionType.Boolean, hasarg=True, cfgFieldName='showProgress'),
]

EXPLAIN_OPTIONS = COMMON_OPTIONS + [ ConfigOpt(longopt='qfile', shortopt='q', desc='file storing aggregation query result', hasarg=True, cfgFieldName='query_result_file'),
                    ConfigOpt(longopt='cfile', shortopt='c', desc='file storing patterns', hasarg=True, cfgFieldName='constraint_file'),
                    ConfigOpt(longopt='ufile', shortopt='u', desc='file storing user question', hasarg=True, cfgFieldName='user_question_file'),
                    ConfigOpt(longopt='ofile', shortopt='o', desc='file to write output to', hasarg=True, cfgFieldName='outfile'),
                    ConfigOpt(longopt='epsilon', shortopt='e', desc='file to write output to', hasarg=True, otype=OptionType.Float, cfgFieldName='constraint_epsilon'),
                    ConfigOpt(longopt='aggcolumn', shortopt='a', desc='column that was input to the aggregation function', hasarg=True, cfgFieldName='aggregate_column'), 
]

STATS_OPTIONS = COMMON_OPTIONS + [ 
]

HELP_OPTIONS = [ ConfigOpt(longopt='log', shortopt='l', desc='select log level {DEBUG,INFO,WARNING,ERROR}', hasarg=True, value="ERROR"),
]

# mapping strings to log levels
LOGLEVELS_MAP = { "DEBUG": logging.DEBUG,
                  "INFO": logging.INFO,
                  "WARNING": logging.WARNING,
                  "ERROR": logging.ERROR,
}

# commands supported by Cape
COMMANDS = [ Command(cmd=CmdTypes.Mine,cmdstr='mine',options=CmdOptions(MINE_OPTIONS),helpMessage='Mining patterns that hold for a relation (necessary preprocessing step for generating explanations.',execute=mineCommand),
             Command(cmd=CmdTypes.Explain,cmdstr='explain',options=CmdOptions(EXPLAIN_OPTIONS),helpMessage='Generate explanations for an aggregation result (patterns should have been mined upfront using mine).', execute=explainCommand),
             Command(cmd=CmdTypes.Stats,cmdstr='stats',options=CmdOptions(STATS_OPTIONS),helpMessage='Extracting statistics from database collected during previous mining executions.',execute=statsCommand),
             Command(cmd=CmdTypes.Help,cmdstr='help',options=CmdOptions(HELP_OPTIONS),helpMessage='Show general or command specific help.', execute=helpCommand),
]

# maps command names to Command objects
COMMAND_BY_TYPE = { x.cmd : x for x in COMMANDS}

# ********************************************************************************
def getCmdList():
    """ 
    Return a help message for the supported commands.
    """
    return "\n".join((x.helpString() for x in COMMANDS))

# ********************************************************************************
def printHelp(c=None):
    """
    Print command specific or general help message.
    """
    if c is None:
        helpMessage = 'capexplain COMMAND [OPTIONS]:\n\texplain unusually high or low aggregation query results.\n\nAVAILABLE COMMANDS:\n\n' + getCmdList()
    elif c.cmd == CmdTypes.Help:
        helpMessage = 'capexplain help [COMMAND]: show general or command specific help message.\n\nAVAILABLE COMMANDS:\n\n' + getCmdList()
    else:
        helpMessage = 'capexplain {} [OPTIONS]:\n\t{}\n\nSUPPORTED OPTIONS:\n{}'.format(
            c.cmdstr,
            c.helpMessage,
            "\n".join(o.helpString() for o in c.options.optionlist))
    print(helpMessage)

def parseOptions(argv):
    """
    Parse options from commandline.
    """
    # detect command
    if len(argv) > 0:
        cmdString = argv[0]
        command = next((c for c in COMMANDS if c.cmdstr == cmdString), None)
        argv=argv[1:]
    else:
        command = None
    if command is None:
        printHelp()
        sys.exit(2)

    if (command.cmd == CmdTypes.Help):
        helpCommand(argv)
        sys.exit(1)
    
    # parse options
    o = command.options
    try:
        opts, args = getopt.getopt(argv, o.shortopts, o.longopts)
    except getopt.GetoptError as e:
        print("Exception {}\n\n{}\n", type(e), e.args)
        printHelp(command)
        sys.exit(2)
        
    # store command options and args
    for opt, arg in opts:
        if opt in o.shortopt_map:
            cmdopt = o.shortopt_map[opt]
            if cmdopt.hasarg:
                cmdopt.value = arg
            else:
                cmdopt.value = True
        elif opt in o.longopt_map:
            cmdopt = o.longopt_map[opt]
            if cmdopt.hasarg:
                cmdopt.value = arg
            else:
                cmdopt.value = True
        else:
            print("invalid option {} for command {}".format(opt, command))
            sys.exit(2)

    # cast options 
    for cmdopt in o.optionlist:
        cmdopt.castValue()

    # print help and exist if user asked for help
    if o.cmdConfig['help'].value is not None:
        printHelp(command)
        sys.exit(2)
        
    # if log level is set then record that
    if "log" in o.cmdConfig:
        loglevel = o.cmdConfig["log"].value
        
    return loglevel, command




def main(argv=sys.argv[1:]):
    """
    cape main function
    """
    loglevel, command = parseOptions(argv)
    logging.basicConfig(level=LOGLEVELS_MAP[loglevel],format=LOGFORMAT)    
    log = logging.getLogger(__name__)
    log.debug("...started")
    log.debug("parsed options")
    log.debug("execute command: {}".format(command.cmdstr))
    command.execute(command,log=log)
    log.debug("Cape is finished")

if __name__=="__main__":
    main(sys.argv[1:])
