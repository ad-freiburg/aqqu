"""
A module for a relation oracle

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging

logger = logging.getLogger(__name__)


class RelationOracle:
    """
    An oracle that returns matching relations for a query.
    This oracle will be set as a member of a query object,
    so the pattern matching process can ask the oracle
    while matching relations.
    """

    def __init__(self, gold_queries):
        # A map from query -> {relations}
        self.query_relations = dict()
        self._build_query_relations(gold_queries)

    def _build_query_relations(self, gold_queries):
        for q in gold_queries:
            sparql = q['targetSparql']
            relations = set()
            # Ugly but works. Only extract the part that contains the relations
            where_clause = sparql[sparql.rindex('WHERE {') + 7:sparql.rindex('}')]
            where_clause = where_clause.strip(' }')
            where_elements = where_clause.split(' . ')
            for we in where_elements:
                if we.startswith('OPTIONAL') or we.startswith('FILTER'):
                    continue
                else:
                    triple = we.split(' ')
                    relation = triple[1]
                    # Remove fb: prefix
                    relation = relation[3:]
                    relations.add(relation)
            self.query_relations[q['utterance']] = relations

    def is_relation_in_query(self, query, relation, reverse_relations):
        if query.query_text in self.query_relations:
            query_rels = self.query_relations[query.query_text]
            reverse_relation = None
            if relation in reverse_relations:
                reverse_relation = reverse_relations[relation]
            if relation in query_rels or reverse_relation in query_rels:
                return True
            else:
                return False
        else:
            logger.warn("Relation Oracle does not know about query %s" % query)
            return False






