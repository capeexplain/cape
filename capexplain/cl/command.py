from enum import Enum, unique
from capexplain.cl.cfgoption import DictLike
import logging

# logger for this module
log = logging.getLogger(__name__)

################################################################################
# Cmd types
@unique
class CmdTypes(Enum):
    Mine = 1,
    Explain = 2,
    Stats = 3,
    Help = 4,
    GUI = 5

# ********************************************************************************
# Information about a command for capexplain


class Command:
    """
    A subcommand for a CLI program with the CL options supported by this subcommand and a help message.
    """

    def __init__(self, cmd, cmdstr, helpMessage, execute, options=None):
        """
        cmd is the name of the command, cmdstr is used to call this command, execute is the function that implements the command, and options is a CmdOptions object describing supported options for this command.
        """
        self.cmd = cmd
        self.cmdstr = cmdstr
        self.options = options
        self.helpMessage = helpMessage
        self.execute = execute

    def helpString(self):
        """
        Returns a help message for the command.
        """
        return '{:30}- {}'.format(self.cmdstr, self.helpMessage)

    def __str__(self):
        return "Command(cmd={0}, cmdstr={1}, helpMessage={2}, execute={3}, options={4})".format(self.cmd, self.cmdstr, self.helpMessage, self.execute, str(self.options))

# ********************************************************************************
# multiple indexes for the options for a command


class CmdOptions:
    """
    Stores the CL options (represented as cape.cl.cfgoption.ConfigOpt objects) supported by a command.
    """

    def constructOptions(self):
        """
        Creates various auxiliary datastructures: a string containing all characters supported as short options (e.g., -a, -b => 'ab'), a list of long options (strings), a map from short option to option object, a map of long options to option object, and a map of configuration option name to options object.
        """
        self.shortopts = ''
        self.longopts = []
        self.shortopt_map = {}
        self.longopt_map = {}
        self.cmdConfig = {}
        for opt in self.optionlist:
            # mapping from configuration option names to configuration objects
            self.cmdConfig[opt.getConfigKey()] = opt
            if opt.shortopt != None:
                # map short option to configuration object
                self.shortopt_map['-' + opt.shortopt] = opt
                self.shortopts += opt.shortopt
                if opt.hasarg:
                    self.shortopts += ':'
            self.longopt_map['--' + opt.longopt] = opt
            if opt.hasarg:
                self.longopts.append(opt.longopt + '=')
            else:
                self.longopts.append(opt.longopt)

    def __init__(self, optionlist):
        self.optionlist = optionlist
        self.shortopt = ''
        self.longopts = ''
        self.longopt_map = {}
        self.shortopt_map = {}
        self.constructOptions()

    def setupConfig(self, config: DictLike):
        """
        Populate config (a DictLike object) based on the options stores in this CmdOptions object.
        """
        o = self
        for opt in o.cmdConfig:
            option = o.cmdConfig[opt]
            if option.value != None:
                key = opt
                val = option.value
                log.debug("option: {}:{}".format(key, val))
                if key in config.getValidKeys():
                    config[key] = val
                else:
                    log.warning(
                        "unhandled config option <{}>".format(str(option)))

    def setupConfigAndConnection(self, conn, config: DictLike):
        """
        Populate config (a DictLike object) based on the options stores in this CmdOptions object and also setup a database connection using the connection parameters provided by this CmdOptions object.
        """
        o = self
        for opt in o.cmdConfig:
            option = o.cmdConfig[opt]
            if option.value != None:
                key = opt if (
                    option.cfgFieldName is None) else option.cfgFieldName
                val = option.value
                # log.debug("option: {}:{}".format(key, val))
                if key in conn.getValidKeys():
                    conn[key] = val
                elif key in config.getValidKeys():
                    config[key] = val
                else:
                    log.warning(
                        "unhandled config option <{}>".format(option.longopt))

    def __str__(self):
        return "Command Options: (\n{0})".format("\n".join(str(x) for x in self.optionlist))
