"""
A module for relation and entity oracles.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging
from entity_linker.entity_linker import IdentifiedEntity, get_value_for_year, \
    DateValue

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


class EntityOracle:
    """
    This class can be used in place of an entity linker.
    It provides the identify_entities_in_tokens method that
    wraps around a standard entity_linker. This is needed to obtain
    information about the identified entities like popularity and readable
    name etc.
    """

    def __init__(self, oracle_entities_file):
        self.tokens_mid_map = dict()
        self._read_oracle_entities(oracle_entities_file)

    def _read_oracle_entities(self, oracle_entities_file):
        with open(oracle_entities_file, 'rb') as f:
            for line in f:
                tokens, mid = line.decode('utf-8').strip().split('\t')
                tokens = tokens.replace(' ', '')
                if tokens not in self.tokens_mid_map:
                    self.tokens_mid_map[tokens] = []
                self.tokens_mid_map[tokens].append(mid)

    def identify_entities_in_tokens(self, tokens, entity_linker):
        logger.info("Using entity oracle...")
        identified_entities = []
        for i in range(1, len(tokens) + 1):
            for j in range(i):
                span = tokens[j:i]
                span_str = ''.join([t.token for t in span])
                if span_str in self.tokens_mid_map:
                    mids = self.tokens_mid_map[span_str]
                    for mid in mids:
                        if mid.startswith('/type/datetime') \
                                or mid.startswith('/un/'):
                            e = DateValue(span_str, get_value_for_year(span_str))
                            ie = IdentifiedEntity(span, e.name, e, perfect_match=True)
                            identified_entities.append(ie)
                        else:
                            entity = entity_linker.get_entity_for_mid(mid)
                            if entity:
                                ie = IdentifiedEntity(span,
                                                      entity.name,
                                                      entity, entity.score,
                                                      1.0,
                                                      perfect_match=True)
                                identified_entities.append(ie)
                            else:
                                logger.warn("Oracle entity does not exist:"
                                            " %s." % mid)
        return identified_entities




