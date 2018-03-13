"""
A module for an entity oracle

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging
from .entity_linker import IdentifiedEntity, get_value_for_year, \
    DateValue

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class EntityOracle:
    """
    This class can be used in place of an entity linker.
    It provides the identify_entities_in_tokens method that
    wraps around a standard entity_linker. This is needed to obtain
    information about the identified entities like popularity and readable
    name etc.
    """

    def __init__(self, oracle_entities_file, entity_index):
        self.tokens_mid_map = dict()
        self.entity_index = entity_index
        self._read_oracle_entities(oracle_entities_file)

    @staticmethod
    def init_from_config(ranker_params, entity_index):
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        return EntityOracle(ranker_params.entity_oracle_file, entity_index)

    def _read_oracle_entities(self, oracle_entities_file):
        with open(oracle_entities_file, 'rb') as f:
            for line in f:
                tokens, mid = line.decode('utf-8').strip().split('\t')
                tokens = tokens.replace(' ', '')
                if tokens not in self.tokens_mid_map:
                    self.tokens_mid_map[tokens] = []
                self.tokens_mid_map[tokens].append(mid)

    def identify_entities_in_tokens(self, tokens):
        logger.info("Using entity oracle...")
        identified_entities = []
        for i in range(1, len(tokens) + 1):
            for j in range(i):
                span = tokens[j:i]
                span_str = ''.join([t.orth_ for t in span])
                if span_str in self.tokens_mid_map:
                    mids = self.tokens_mid_map[span_str]
                    for mid in mids:
                        if mid.startswith('/type/datetime') \
                                or mid.startswith('/un/'):
                            e = DateValue(span_str, get_value_for_year(span_str))
                            ie = IdentifiedEntity(span, e.name, e,
                                                  types=['Date'],
                                                  category='Date',
                                                  perfect_match=True)
                            identified_entities.append(ie)
                        else:
                            entity = self.entity_index.get_entity_for_mid(mid)
                            if entity:
                                types = self.entity_index.get_types_for_mid(mid, 1)
                                category = self.entity_index.get_category_for_mid(mid)
                                ie = IdentifiedEntity(span,
                                                      entity.name,
                                                      entity,
                                                      types=types,
                                                      category=category,
                                                      score=entity.score,
                                                      surface_score=1.0,
                                                      perfect_match=True)
                                identified_entities.append(ie)
                            else:
                                logger.warn("Oracle entity does not exist:"
                                            " %s." % mid)
        return identified_entities, []  # no text entities
