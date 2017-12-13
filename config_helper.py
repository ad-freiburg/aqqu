"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from configparser import ConfigParser
import logging

logger = logging.getLogger(__name__)

# When a configuration is read, store it here to make it accessible.
config = None

def read_configuration(configfile):
    """Read configuration and set variables.

    :return:
    """
    global config
    logger.info("Reading configuration from: " + configfile)
    parser = ConfigParser()
    parser.read(configfile)
    config = parser
    return parser
