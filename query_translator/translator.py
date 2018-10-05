"""
A module for simple query translation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging
import time
from typing import Tuple, Iterable
import spacy
import sparql_backend.loader
import config_helper
from answer_type.answer_type_identifier import AnswerTypeIdentifier
from entity_linker.entity_index_rocksdb import EntityIndex
from query_translator.pattern_matcher import QueryCandidateExtender,\
        QueryPatternMatcher, get_content_tokens
from query_translator import ranker

logger = logging.getLogger(__name__)

class Query:
    """
    A query that is to be translated.
    """

    def __init__(self, query_doc):
        self.tokens = query_doc  # type: spacy.tokens.Doc
        self.text = query_doc.text.lower()  # type: str
        self.target_type = None  # type: AnswerType
        self.content_tokens = None  # type: spacy.tokens.Span
        self.identified_entities = None  # type: Iterable[IdentifiedEntity]
        self.text_entities = None  # type: Iterable[IdentifiedEntity]
        self.relation_oracle = None
        self.is_count_query = False
        # self.previous_entities = None

class QueryTranslator(object):

    def __init__(self, backend,
                 query_extender,
                 entity_linker,
                 nlp,
                 scorer,
                 entity_index,
                 answer_type_identifier):
        self.backend = backend
        self.query_extender = query_extender
        self.entity_linker = entity_linker
        self.nlp = nlp
        self.scorer = scorer
        self.entity_index = entity_index
        self.answer_type_identifier = answer_type_identifier
        self.query_extender.set_parameters(scorer.get_parameters())

    @staticmethod
    def init_from_config():
        config_params = config_helper.config
        backend_module_name = config_params.get("Backend", "backend")
        backend = sparql_backend.loader.get_backend(backend_module_name)
        query_extender = QueryCandidateExtender.init_from_config()
        nlp = spacy.load('en_core_web_lg')
        scorer = ranker.SimpleScoreRanker('DefaultScorer')
        entity_index = EntityIndex.init_from_config()
        entity_linker = scorer.parameters.\
            entity_linker_class.init_from_config(
                scorer.get_parameters(),
                entity_index)
        answer_type_identifier = AnswerTypeIdentifier.init_from_config()
        return QueryTranslator(backend, query_extender,
                               entity_linker, nlp, scorer, entity_index,
                               answer_type_identifier)

    def set_ranker(self, scorer):
        """Sets the parameters of the translator.

        :type ranker: ranker.Ranker
        :return:
        """
        self.scorer = scorer
        params = scorer.get_parameters()
        if type(self.entity_linker) != params.entity_linker_class:
            self.entity_linker = params.entity_linker_class.init_from_config(
                            params,
                            self.entity_index)

        self.query_extender.set_parameters(params)

    def get_ranker(self):
        """Returns the current parameters of the translator.
        """
        return self.scorer

    def translate_query(self, query_text):
        """
        Perform the actual translation.
        :param query_text:
        :param relation_oracle:
        :return:
        """
        # Parse query.
        logger.info("Translating query: %s." % query_text)
        start_time = time.time()
        # Parse the query.
        query = self.parse_and_identify_entities(query_text)
        # Identify the target type.
        self.answer_type_identifier.identify_target(query)
        # Set the relation oracle.
        query.relation_oracle = self.scorer.get_parameters().relation_oracle
        # Get content tokens of the query.
        query.content_tokens = get_content_tokens(query.tokens)
        # Match the patterns.
        pattern_matcher = QueryPatternMatcher(query,
                                              self.query_extender,
                                              self.backend)
        ert_matches = []
        ermrt_matches = []
        ermrert_matches = []
        ert_matches = pattern_matcher.match_ERT_pattern()
        ermrt_matches = pattern_matcher.match_ERMRT_pattern()
        ermrert_matches = pattern_matcher.match_ERMRERT_pattern()
        duration = (time.time() - start_time) * 1000
        logging.info("Total translation time: %.2f ms." % duration)
        return query, ert_matches + ermrt_matches + ermrert_matches

    def parse_and_identify_entities(self, query_text):
        """
        Parses the provided text, identifies entities and a list with
        previous identified entities.
        Returns a query object.
        :param query_text:
        :return:
        """
        # Parse query.
        query_doc = self.nlp(query_text)
        # Create a query object.
        query = Query(query_doc)

        """ entities is a list of objects,
        text_entities is always None"""

        entities, text_entities = self.entity_linker.identify_entities_in_tokens(
            query.tokens)

        query.identified_entities = entities
        query.text_entities = text_entities
        return query

    def translate_and_execute_query(self, query, n_top=200):
        """
        Translates the query and returns a list
        of type TranslationResult.
        :param query:
        :return:
        """
        # Parse query.
        num_sparql_queries = self.backend.num_queries_executed
        sparql_query_time = self.backend.total_query_time
        parsed_query, query_candidates = self.translate_query(query)
        translation_time = (self.backend.total_query_time - sparql_query_time) * 1000
        num_sparql_queries = self.backend.num_queries_executed - num_sparql_queries
        avg_query_time = translation_time / (num_sparql_queries + 0.001)
        logger.info("Translation executed %s queries in %.2f ms."
                    " Average: %.2f ms." % (num_sparql_queries,
                                            translation_time, avg_query_time))
        logger.info("Ranking %s query candidates" % len(query_candidates))
        ranker = self.scorer
        ranked_candidates = ranker.rank_query_candidates(query_candidates,
                                                         store_features=True)
        sparql_query_time = self.backend.total_query_time
        logger.info("Fetching results for all candidates.")
        n_total_answers = 0
        if len(ranked_candidates) > n_top:
            logger.info("Truncating returned candidates to %s." % n_top)
        for query_candidate in ranked_candidates[:n_top]:
            query_candidate.retrieve_result(include_name=True)
            n_total_answers += sum(
                [len(rows) for rows in query_candidate.query_result])
        # This assumes that each query candidate uses the same SPARQL backend
        # instance which should be the case at the moment.
        result_fetch_time = (self.backend.total_query_time - sparql_query_time) * 1000
        avg_result_fetch_time = result_fetch_time / (len(ranked_candidates[:n_top]) + 0.001)
        logger.info("Fetched a total of %s results in %s queries in %.2f ms."
                " Avg per query: %.2f ms." % (n_total_answers, len(ranked_candidates[:n_top]),
                                                  result_fetch_time, avg_result_fetch_time))
        logger.info("Done translating and executing: %s." % query)
        return parsed_query, ranked_candidates[:n_top]




if __name__ == '__main__':
    logger.warn("No MAIN")

