# Copyright (c) 2017 University of Freiburg
# Chair of Algorithms and Data Structures
# Author: Niklas Schnelle

import importlib
import logging
import globals

logger = logging.getLogger(__name__)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(sparql_backend = {})
def get_backend(backend_module_name):
    logger.info("Loading backend: {}".format(backend_module_name))

    backend_module = importlib.import_module('.'+backend_module_name, 'sparql_backend')
    Backend = getattr(backend_module, 'Backend')
    if backend_module_name not in get_backend.sparql_backend:
        config_options = globals.config
        get_backend.sparql_backend[backend_module_name] = \
                Backend.init_from_config(config_options)
    return get_backend.sparql_backend[backend_module_name]
