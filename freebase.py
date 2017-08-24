"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""

FREEBASE_NS_PREFIX = "http://rdf.freebase.com/ns/"
FREEBASE_NS_PREFIX_BYTES = FREEBASE_NS_PREFIX.encode('utf-8')
FREEBASE_SPARQL_PREFIX = "fb"
FREEBASE_NAME_RELATION = "type.object.name"
FREEBASE_KEY_PREFIX = "http://rdf.freebase.com/key/"


def get_prefixed_qualified_mid(mid, prefix):
    return "%s:%s" % (prefix, mid)

def get_qualified_mid(mid):
    '''
    Returns a fully qualified MID, with NS
    prefix and brackets.
    :param mid:
    :return:
    '''
    return "<%s%s>" % (FREEBASE_NS_PREFIX, mid)

def get_mid_from_qualified_string(qualified_str):
    '''
    Returns a mid from a fully qualified string
    :param qualified_str
    :return:
    '''
    if qualified_str.startswith('<') and qualified_str.endswith('>'):
        qualified_str = qualified_str[1:-1]
    return remove_freebase_ns(qualified_str)

def remove_freebase_ns(mid):
    '''
    Returns a MID without freebase namespace
    :param mid:
    :return:
    '''
    if mid.startswith(FREEBASE_NS_PREFIX):
        return mid[len(FREEBASE_NS_PREFIX):]
    return mid

def remove_freebase_ns_bytes(mid):
    '''
    Returns a MID without freebase namespace
    :param mid:
    :return:
    '''
    if mid.startswith(FREEBASE_NS_PREFIX_BYTES):
        return mid[len(FREEBASE_NS_PREFIX_BYTES):]
    return mid

