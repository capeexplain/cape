from enum import Enum, unique

# ********************************************************************************
# commandline option possible datatypes
class OptionType(Enum):
    Int = 'int',
    Float = 'float',
    String = 'string',
    Boolean = 'boolean'

# ********************************************************************************
# a commandline option
class ConfigOpt:
    
    def __init__(self, longopt, shortopt=None, desc=None, hasarg=False, value=None, otype=OptionType.String, cfgFieldName=None):
        self.longopt = longopt
        self.shortopt = shortopt
        self.hasarg = hasarg
        self.description = desc
        self.value = value
        self.otype=otype
        self.cfgFieldName = cfgFieldName

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

    def castValue(self):
        if self.hasarg == True and self.value is not None:
            if self.otype == OptionType.Int:
                self.value =  int(self.value)
            elif self.otype == OptionType.Float:
                self.value =  float(self.value)
            elif self.otype == OptionType.Boolean:
                self.value =  int(self.boolean)
            return self
