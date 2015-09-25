"""
A module for simple query translation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
from entity_linker.entity_linker import EntityLinker
from answer_type import AnswerTypeIdentifier
from pattern_matcher import QueryCandidateExtender, QueryPatternMatcher, get_content_tokens
import logging
import ranker
import time
from corenlp_parser.parser import CoreNLPParser
import globals
import collections

logger = logging.getLogger(__name__)

class Query:
    """
    A query that is to be translated.
    """

    def __init__(self, text):
        self.query_text = text.lower()
        self.original_query = self.query_text
        self.target_type = None
        self.query_tokens = None
        self.query_content_tokens = None
        self.identified_entities = None
        self.consistent_entities = None
        self.relation_oracle = None
        self.is_count_query = False
        self.transform_query(self.query_text)

    def transform_query(self, text):
        """
        For some questions we want to transform the query text
        before processing, e.g. when asking "how many ..." queries.
        """
        how_many = "how many"
        in_how_many = "in how many"
        if text.startswith(how_many):
            self.is_count_query = True
            # Replace "how many" with "what"
            self.query = "what" + text[len(how_many):]
            self.original_query = text
        elif text.startswith(in_how_many):
            self.is_count_query = True
            # Replace "how many" with "what"
            self.query = "in what" + text[len(in_how_many):]
            self.original_query = text



class QueryTranslator(object):

    def __init__(self, sparql_backend,
                 query_extender,
                 entity_linker,
                 parser,
                 scorer_obj):
        self.sparql_backend = sparql_backend
        self.query_extender = query_extender
        self.entity_linker = entity_linker
        self.parser = parser
        self.scorer = scorer_obj
        self.query_extender.set_parameters(scorer_obj.get_parameters())

    @staticmethod
    def init_from_config():
        config_params = globals.config
        sparql_backend = globals.get_sparql_backend(config_params)
        query_extender = QueryCandidateExtender.init_from_config()
        entity_linker = EntityLinker.init_from_config()
        parser = CoreNLPParser.init_from_config()
        scorer_obj = ranker.SimpleScoreRanker('DefaultScorer')
        return QueryTranslator(sparql_backend, query_extender,
                               entity_linker, parser, scorer_obj)

    def set_scorer(self, scorer):
        """Sets the parameters of the translator.

        :type scorer: ranker.BaseRanker
        :return:
        """
        # TODO(Elmar): should this be a parameter of a function call?
        self.scorer = scorer
        self.query_extender.set_parameters(scorer.get_parameters())

    def get_scorer(self):
        """Returns the current parameters of the translator.
        """
        return self.scorer

    def translate_query(self, query_text):
        """
        Perform the actual translation.
        :param query_text:
        :param relation_oracle:
        :param entity_oracle:
        :return:
        """
        # Parse query.
        logger.info("Translating query: %s." % query_text)
        start_time = time.time()
        # Parse the query.
        query = self.parse_and_identify_entities(query_text)
        # Set the relation oracle.
        query.relation_oracle = self.scorer.get_parameters().relation_oracle
        # Identify the target type.
        target_identifier = AnswerTypeIdentifier()
        target_identifier.identify_target(query)
        # Get content tokens of the query.
        query.query_content_tokens = get_content_tokens(query.query_tokens)
        # Match the patterns.
        pattern_matcher = QueryPatternMatcher(query,
                                              self.query_extender,
                                              self.sparql_backend)
        ert_matches = []
        ermrt_matches = []
        ermrert_matches = []
        ert_matches = pattern_matcher.match_ERT_pattern()
        ermrt_matches = pattern_matcher.match_ERMRT_pattern()
        ermrert_matches = pattern_matcher.match_ERMRERT_pattern()
        duration = (time.time() - start_time) * 1000
        logging.info("Total translation time: %.2f ms." % duration)
        return ert_matches + ermrt_matches + ermrert_matches

    def parse_and_identify_entities(self, query_text):
        """
        Parses the provided text and identifies entities.
        Returns a query object.
        :param query_text:
        :param entity_oracle:
        :return:
        """
        # Parse query.
        parse_result = self.parser.parse(query_text)
        tokens = parse_result.tokens
        # Create a query object.
        query = Query(query_text)
        query.query_tokens = tokens
        if not self.scorer.get_parameters().entity_oracle:
            entities = self.entity_linker.identify_entities_in_tokens(
                query.query_tokens)
        else:
            entity_oracle = self.scorer.get_parameters().entity_oracle
            entities = entity_oracle.identify_entities_in_tokens(
                query.query_tokens,
                self.entity_linker)
        query.identified_entities = entities
        return query

    def translate_and_execute_query(self, query, n_top=200):
        """
        Translates the query and returns a list
        of namedtuples of type TranslationResult.
        :param query:
        :return:
        """
        TranslationResult = collections.namedtuple('TranslationResult',
                                                   ['query_candidate',
                                                    'query_result_rows'],
                                                   verbose=False)
        # Parse query.
        results = []
        num_sparql_queries = self.sparql_backend.num_queries_executed
        sparql_query_time = self.sparql_backend.total_query_time
        queries_candidates = self.translate_query(query)
        translation_time = (self.sparql_backend.total_query_time - sparql_query_time) * 1000
        num_sparql_queries = self.sparql_backend.num_queries_executed - num_sparql_queries
        avg_query_time = translation_time / (num_sparql_queries + 0.001)
        logger.info("Translation executed %s queries in %.2f ms."
                    " Average: %.2f ms." % (num_sparql_queries,
                                            translation_time, avg_query_time))
        logger.info("Ranking %s query candidates" % len(queries_candidates))
        ranker = self.scorer
        ranked_candidates = ranker.rank_query_candidates(queries_candidates)
        logger.info("Fetching results for all candidates.")
        sparql_query_time = self.sparql_backend.total_query_time
        n_total_results = 0
        if len(ranked_candidates) > n_top:
            logger.info("Truncating returned candidates to %s." % n_top)
        for query_candidate in ranked_candidates[:n_top]:
            query_result = query_candidate.get_result(include_name=True)
            n_total_results += sum([len(rows) for rows in query_result])
            result = TranslationResult(query_candidate, query_result)
            results.append(result)
        # This assumes that each query candidate uses the same SPARQL backend
        # instance which should be the case at the moment.
        result_fetch_time = (self.sparql_backend.total_query_time - sparql_query_time) * 1000
        avg_result_fetch_time = result_fetch_time / (len(results) + 0.001)
        logger.info("Fetched a total of %s results in %s queries in %.2f ms."
                    " Avg per query: %.2f ms." % (n_total_results, len(results),
                                                  result_fetch_time, avg_result_fetch_time))
        logger.info("Done translating and executing: %s." % query)
        return results


class TranslatorParameters(object):

    """A class that holds parameters for the translator."""
    def __init__(self):
        self.entity_oracle = None
        self.relation_oracle = None
        # When generating candidates, restrict them to the
        # deterimined answer type.
        self.restrict_answer_type = True
        # When matching candidates, require that relations
        # match in some way in the question.
        self.require_relation_match = True


def get_suffix_for_params(parameters):
    """Return a suffix string for the selected parameters.

    :type parameters TranslatorParameters
    :param parameters:
    :return:
    """
    suffix = ""
    if parameters.entity_oracle:
        suffix += "_eo"
    if not parameters.require_relation_match:
        suffix += "_arm"
    if not parameters.restrict_answer_type:
        suffix += "_atm"
    return suffix


if __name__ == '__main__':
    logger.warn("No MAIN")

