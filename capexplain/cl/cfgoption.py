from enum import Enum, unique

# ********************************************************************************
# commandline option possible datatypes
@unique
class OptionType(Enum):
    """
    Type of the value of a configuration option
    """
    Int = 'int',
    Float = 'float',
    String = 'string',
    Boolean = 'boolean'

# ********************************************************************************
# a commandline option


class ConfigOpt:
    """
    A configuration option which can be passed on the commandline.
    """

    def __init__(self, longopt, shortopt=None, desc=None, hasarg=False, otype=OptionType.String, cfgFieldName=None, defaultValue=None):
        self.longopt = longopt
        self.shortopt = shortopt
        self.hasarg = hasarg
        self.description = desc
        self.otype = otype
        self.cfgFieldName = cfgFieldName
        self.defaultValue = defaultValue
        if (defaultValue != None):
            self.value = defaultValue
        else:
            self.value = None

    def helpString(self):
        """
        Return a help string for this option (used to create help messages for a commandline program.
        """
        helpMessage = ''
        if self.shortopt != None:
            helpMessage = '-' + self.shortopt + ' ,'
        helpMessage += '--' + self.longopt
        if self.hasarg == True:
            helpMessage += " <arg>"
        if self.description != None:
            helpMessage = '{:30}'.format(helpMessage)
            helpMessage += " - " + self.description
        if self.defaultValue != None:
            helpMessage += " (DEFAULT: " + str(self.defaultValue) + ")"
        return helpMessage

    def castValue(self):
        """
        The value has been set as a string, but this option is of a non-string type, then cast the string to the options type.
        """
        if self.hasarg == True and self.value is not None and type(self.value) == str:
            if self.otype == OptionType.Int:
                self.value = int(self.value)
            elif self.otype == OptionType.Float:
                self.value = float(self.value)
            elif self.otype == OptionType.Boolean:
                self.value = {'True': True,
                              'True': True,
                              't': True,
                              '1': True,
                              'False': False,
                              'false': False,
                              'f': False,
                              '0': False}[self.value]
                self.value = bool(self.value)
            return self

    def getConfigKey(self):
        """
        Get the key of the configuration this option is stored at (either cftFieldName if it exists or longopt otherwhise)
        """
        return self.longopt if(self.cfgFieldName is None) else self.cfgFieldName

    def __str__(self):
        return "ConfigOption(longopt={0}, shortopt={1}, hasarg={2}, description={3}, otype={4}, cfgFieldName={5}, defaultValue={6}, value={7})".format(self.longopt, self.shortopt, self.hasarg, self.description, self.otype, self.cfgFieldName, self.defaultValue, self.value)

# ********************************************************************************
# add support for accessing a class like a dictionary.


class DictLike:
    """
    Class used like an interface that makes a configuration class (data object) behave like a dictionary.
    """
    # overwrite __getitem__ to allow dictory style access to options

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise AttributeError("No such attribute: " + key)
        return self.__dict__[key]

    # overwrite __setitem__ to allow dictory style setting of options
    def __setitem__(self, key, value):
        if key not in self.__dict__:
            raise AttributeError("No such attribute: " + key)
        self.__dict__[key] = value

    # get access to list of field names
    def getValidKeys(self):
        return self.__dict__

    # check whether configuration is valid
    def validateConfiguration(self):
        return True
