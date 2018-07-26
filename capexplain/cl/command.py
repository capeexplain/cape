from enum import Enum, unique

################################################################################
# Cmd types
@unique
class CmdTypes(Enum):
    Mine = 1,
    Explain = 2,
    Stats = 3,
    Help = 4

# ********************************************************************************
# Information about a command for capexplain
class Command:

    def __init__(self, cmd, cmdstr, helpMessage, execute, options=None):
        self.cmd = cmd
        self.cmdstr = cmdstr
        self.options=options
        self.helpMessage=helpMessage
        self.execute = execute

    def helpString(self):
        return '{:30}- {}'.format(self.cmdstr,self.helpMessage)

    def __str__(self):
        return self.__dict__.__str__()

# ********************************************************************************
# multiple indexes for the options for a command
class CmdOptions:

    def constructOptions(self):
        self.shortopts = ''
        self.longopts = []
        self.shopt_map = {}
        self.longopt_map = {}
        self.cmdConfig = {}
        for opt in self.optionlist:
            self.cmdConfig[opt.longopt] = opt # mapping from configuration option names to configuration objects
            if opt.shortopt != None:
                self.shortopt_map['-' + opt.shortopt] = opt # map short option to configuration object
                self.shortopts+=opt.shortopt
                if opt.hasarg:
                    self.shortopts+=':'
            self.longopt_map['--' + opt.longopt] = opt
            if opt.hasarg:
                self.longopts.append(opt.longopt + '=')
            else:
                self.longopts.append(opt.longopt)

    def __init__(self, optionlist):
        self.optionlist = optionlist
        self.shortopt=''
        self.longopts=''
        self.longopt_map = {}
        self.shortopt_map = {}
        self.constructOptions()
