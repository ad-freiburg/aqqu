"""
A module for identifying entities which are closely related to the query
string, either by being mentioned in the query or via a combined SPARQL + Text
query against a hybrid knowledge base. Uses a standard EntityLinker to identify
directly mentioned entities and combines it with QLever SPARQL + Text queries
to identify entities mentioned in large text corpus and with a context similar
to the question.

Copyright 2017, University of Freiburg.

Niklas Schnelle <schnelle@informatik.uni-freiburg.de>
"""
import logging
import re
import time
import globals
from .surface_index_memory import EntitySurfaceIndexMemory
from .entity_linker import EntityLinker, Entity, KBEntity, Value, DateValue,\
                            IdentifiedEntity
import sparql_backend.loader
from corenlp_parser.parser import Token

logger = logging.getLogger(__name__)

class EntityLinkerQlever(EntityLinker):

    def __init__(self, surface_index, qlever_backend, stopwords,
            max_entities_per_tokens=4, max_text_entities = 5):
        super().__init__(surface_index, max_entities_per_tokens)
        self.qlever_backend = qlever_backend
        self.stopwords = stopwords
        self.max_text_entities = max_text_entities
        self.min_subrange = 3

    @staticmethod
    def init_from_config(ranker_params):
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = globals.config
        stopwords = EntityLinkerQlever.load_stopwords(
                config_options.get('EntityLinkerQlever',
                'stopwords'))
        surface_index = EntitySurfaceIndexMemory.init_from_config()
        max_entities_per_tokens = int(config_options.get('EntityLinker',
                                                      'max-entites-per-tokens'))
        qlever_backend = sparql_backend.loader.get_backend('qlever')
        return EntityLinkerQlever(surface_index, qlever_backend, stopwords,
                max_entities_per_tokens=max_entities_per_tokens)

    def textEntityQuery(self, tokens, limit):
        toks_nostop = [t for t in tokens 
                if t.token.lower() not in self.stopwords]
        entities = []
        max_start = min(len(toks_nostop), max(0, len(toks_nostop)-self.min_subrange))
        for start in range(max_start+1):
            min_subrange_end = min(start + self.min_subrange, len(toks_nostop))
            for end in range(min_subrange_end, len(toks_nostop)+1):
                new_entities = self.simpleTextEntityQuery(toks_nostop[start:end], limit)
                entities.extend(new_entities)
        return entities


    def simpleTextEntityQuery(self, tokens, limit):
        text = ' '.join([t.token for t in tokens])
        text_query = """
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?0e ?1ename SCORE(?t) WHERE {{
        ?t <in-text> "{0}" .
        ?0e <in-text> ?t .
        ?0e fb:type.object.name ?1ename .
        }}
        ORDER BY DESC(SCORE(?t))
        LIMIT {1}
        """.format(text, limit)

        logger.info("Text Query: "+text_query)
        results = self.qlever_backend.query(text_query)
        entities = []
        for row in results:
            kbe = KBEntity(row[1], row[0], int(row[2]), [])
            ie = IdentifiedEntity([], # TODO(schnelle) not as in EntityLinker
                                  kbe.name, kbe, kbe.score, 1,
                                  False, text_query = True)
            entities.append(ie)
        return entities

    @staticmethod
    def load_stopwords(stopwordsfile):
        stopwords = set()
        with open(stopwordsfile, 'rt', encoding='utf-8') as swfile:
            for word in swfile:
                stopwords.add(word.strip())
        return stopwords


    def identify_entities_in_tokens(self, tokens, min_surface_score=0.1):
        '''
        Identify instances in the tokens.
        :param tokens: A list of string tokens.
        :return: A list of IdentifiedEntity
        '''
        entities = super().identify_entities_in_tokens(
                tokens, min_surface_score)

        text_entities = self.textEntityQuery(tokens, self.max_text_entities)
        text_entity_map = {te.entity.id: te for te in text_entities}
        for entity in entities:
            if hasattr(entity.entity, 'id') and \
                    entity.entity.id in text_entity_map:
                entity.score += text_entity_map[entity.entity.id].score
                entity.surface_score *= 10
                entity.text_match = True

        return entities


def main():
    globals.read_configuration('config.cfg')
    backend = sparql_backend.loader.get_backend('qlever')

    elq = EntityLinkerQlever.init_from_config()
    query = [('what', 'WP'), ('\'s', 'VBZ'), ('the', 'DT'), ('fastest', 'JJS'), 
            ('airplane', 'NN')]
    tokens = [Token(t, pos) for t, pos in query]


    identified = elq.identify_entities_in_tokens(tokens)
    print('Tokens:', query)
    print([ie.sparql_name() for ie in identified])

if __name__ == '__main__':
    main()
