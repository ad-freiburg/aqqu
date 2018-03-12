#!/usr/bin/env python3
"""
 Implements a SPARQL backend using the QLever API

 Copyright (c) 2017 University of Freiburg
 Chair of Algorithms and Data Structures
 Author: Niklas Schnelle

"""

import logging
import json
import time
from operator import itemgetter
from urllib3 import HTTPConnectionPool, Retry, make_headers
import freebase


logger = logging.getLogger(__name__)

def normalize_freebase_output(text):
    """Remove starting and ending quotes and the namespace prefix.

    :param text:
    :return:
    """
    startQuote = text.find('"')
    if startQuote >= 0 and startQuote + 1 < len(text):
        endQuote = text.rfind('"')
        if endQuote >= 0:
            text = text[startQuote+1:endQuote]

    if len(text) > 1 and text.startswith('<') and text.endswith('>'):
        text = text[1:-1]
    return freebase.remove_freebase_ns(text)


def filter_results_language(results, language):
    """Remove results that contain a literal with another language.

    Empty language is allowed!
    :param results:
    :param language:
    :return:
    """
    langsuffix = "@"+language
    filtered_results = []
    for row in results:
        contains_literal = False
        for value in row:
            value = value.strip()
            if value.startswith('"'):
                contains_literal = True
                if value.endswith(langsuffix):
                    filtered_results.append(row)
        if not contains_literal:
            filtered_results.append(row)
    return filtered_results


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
        self.supports_text = True
        self.lang_in_relations = lang_in_relations
        self.query_log = open('qlever_log.txt', 'wt', encoding='UTF-8')

    def __delete__(self):
        self.query_log.close()

    def _init_connection_pool(self, pool_maxsize, retry=None):
        if not retry:
            # By default, retry on 404 and 503 messages because
            # these seem to happen sometimes, but very rarely.
            retry = Retry(total=10, status_forcelist=[404, 503],
                          backoff_factor=0.1)
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
                lang_in_relations = backend_lir)

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
            logger.debug("Return result from cache! %s", query)
            return self.cache[query]
        start = time.time()
        resp = self.connection_pool.request(method,
                                            self.backend_url,
                                            headers=make_headers(keep_alive=False),
                                            fields=params)
        query_time = (time.time() - start)
        self.total_query_time += query_time
        self.num_queries_executed += 1

        self.query_log.write("----\n{}\n# TOOK {} ms\n".format(query, query_time*1000.0))

        try:
            if resp.status == 200:
                data = json.loads(resp.data.decode('utf-8'))
                key_indices = sorted([(column, key.lstrip('?')) for column, key 
                    in enumerate(data['selected'])], key=itemgetter(1))
                result_rows = data['res']
                if filter_lang:
                    result_rows = filter_results_language(result_rows, filter_lang)
                results = [[normalize_output(row[index]) for index, _ in key_indices]
                        for row in result_rows]
            else:
                logger.warn("Return code %s for query '%s'" % (resp.status,
                                                               query))
                logger.warn("Message: %s" % resp.data)
                results = None
        except ValueError:
            logger.warning("Error executing query: %s.", query)
            logger.warning("Headers: %s.", resp.headers)
            logger.warning("Data: %s.", resp.data)
            results = None
        # Add result to cache.
        if self.cache_enabled and results:
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
     ?s fb:type.object.name "Albert Einstein"@en .
     ?s ?p ?o .
     ?o fb:type.object.name ?x . }
    '''
    print(sparql.query(query))


if __name__ == '__main__':
    main()
