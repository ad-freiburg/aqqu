"""
 Implements a SPARQL backend using the QLever API

 Copyright (c) 2017 University of Freiburg
 Chair of Algorithms and Data Structures
 Author: Niklas Schnelle

"""

from urllib3 import HTTPConnectionPool, Retry
import logging
import globals
import io
import json
import time
from operator import itemgetter

logger = logging.getLogger(__name__)

def normalize_freebase_output(text):
    """Remove starting and ending quotes and the namespace prefix.

    :param text:
    :return:
    """
    if len(text) > 1 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if len(text) > 1 and text.startswith('<') and text.endswith('>'):
        text = text[1:-1]
    return globals.remove_freebase_ns(text)


class Backend(object):
    def __init__(self, backend_host,
                 backend_port,
                 backend_url,
                 # TODO(schnelle) increase once QLever multithreads
                 connection_pool_maxsize=1, 
                 cache_enabled=False,
                 cache_maxsize=10000,
                 retry=None,
                 lang_in_relations=False):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.backend_url = backend_url
        self.connection_pool = None
        self._init_connection_pool(connection_pool_maxsize,
                                   retry=retry)

        # Caching structures.
        self.cache_enabled = cache_enabled
        self.cache_maxsize = cache_maxsize
        self.cache = {}
        self.cached_elements_fifo = []
        self.num_queries_executed = 0
        self.total_query_time = 0.0
        # Backend capabilities
        self.supports_count = False
        self.supports_optional = False
        self.supports_filter = True
        self.lang_in_relations = lang_in_relations

    def _init_connection_pool(self, pool_maxsize, retry=None):
        if not retry:
            # By default, retry on 404 and 503 messages because
            # these seem to happen sometimes, but very rarely.
            retry = Retry(total=10, status_forcelist=[404, 503],
                          backoff_factor=0.2)
        self.connection_pool = HTTPConnectionPool(self.backend_host,
                                                  port=self.backend_port,
                                                  maxsize=pool_maxsize,
                                                  retries=retry)

    @staticmethod
    def init_from_config(config_options):
        """Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """

        backend_host = config_options.get('QLeverBackend', 'backend-host')
        backend_port = config_options.get('QLeverBackend', 'backend-port')
        backend_url = config_options.get('QLeverBackend', 'backend-url')
        backend_lir = config_options.getboolean('QLeverBackend', 'lang-in-relations')
        logger.info("Using QLever SPARQL backend at %s:%s%s" % (
            backend_host, backend_port, backend_url
        ))
        return Backend(backend_host, backend_port, backend_url, 
                retry = 10,  lang_in_relations = backend_lir)

    def query(self, query, method='GET',
                   normalize_output=normalize_freebase_output,
                   filter_lang='en'):
        """Returns the result table of the query as a list of rows.

        :param query:
        :return:
        """
        params = {
            "query": query,
        }
        if self.cache_enabled and query in self.cache:
            logger.debug("Return result from cache! %s" % query)
            return self.cache[query]
        start = time.time()
        resp = self.connection_pool.request(method,
                                            self.backend_url,
                                            fields=params)
        self.total_query_time += (time.time() - start)
        self.num_queries_executed += 1

        try:
            if resp.status == 200:
                data = json.loads(resp.data.decode('utf-8'))
                key_indices = sorted([(column, key.lstrip('?')) for column, key 
                    in enumerate(data['selected'])], key=itemgetter(1))
                result_rows = data['res']
                results = [[normalize_output(row[index]) for index, _ in key_indices]
                        for row in result_rows]
            else:
                logger.warn("Return code %s for query '%s'" % (resp.status,
                                                               query))
                logger.warn("Message: %s" % resp.data)
                results = None
        except ValueError:
            logger.warn("Error executing query: %s." % query)
            logger.warn(traceback.format_exc())
            logger.warn("Headers: %s." % resp.headers)
            logger.warn("Data: %s." % resp.data)
            results = None
        # Add result to cache.
        if self.cache_enabled and results != None:
            self._add_result_to_cache(query, results)
        logger.debug("Processed Result {}".format(results))
        return results

    def _add_result_to_cache(self, query, result):
        self.cached_elements_fifo.append(query)
        self.cache[query] = result
        if len(self.cached_elements_fifo) > self.cache_maxsize:
            to_delete = self.cached_elements_fifo.pop(0)
            del self.cache[to_delete]


def main():
    print("Connecting")
    sparql = Backend('vulcano', '7001', '/')
    query = '''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x
    WHERE {
     ?s fb:type.object.name "Albert Einstein" .
     ?s ?p ?o .
     ?o fb:type.object.name ?x . }
    '''
    print(sparql.query(query))

if __name__ == '__main__':
    main()
