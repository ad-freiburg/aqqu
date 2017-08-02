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
from .entity_linker import EntityLinker, Entity, KBEntity, Value, DateValue,\
                            IdentifiedEntity
import sparql_backend.loader
from corenlp_parser.parser import Token

logger = logging.getLogger(__name__)

class EntityLinkerQlever:

    def __init__(self, sub_linker, qlever_backend, stopwords, max_text_entities = 4):
        self.sub_linker = sub_linker
        self.qlever_backend = qlever_backend
        self.stopwords = stopwords
        self.max_text_entities = max_text_entities
        self.min_subrange = 3

    def textEntityQuery(self, tokens):
        toks_nostop = [t for t in tokens 
                if t.token.lower() not in self.stopwords]
        entities = []
        max_start = min(len(toks_nostop), max(0, len(toks_nostop)-self.min_subrange))
        for start in range(max_start+1):
            min_subrange_end = min(start + self.min_subrange, len(toks_nostop))
            for end in range(min_subrange_end, len(toks_nostop)+1):
                new_entities = self.simpleTextEntityQuery(toks_nostop[start:end])
                logger.info('Found {} entities'.format(len(new_entities)))
                entities.extend(new_entities)
        best = sorted(entities, key = lambda x: x.score, reverse = True)
        return best[:self.max_text_entities]


    def simpleTextEntityQuery(self, tokens):
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
        """.format(text, self.max_text_entities)

        logger.info("Text Query: "+text_query)
        results = self.qlever_backend.query(text_query)
        entities = []
        for row in results:
            kbe = KBEntity(row[1], row[0], int(row[2]), [])
            ie = IdentifiedEntity(tokens, # TODO(schnelle) not as in EntityLinker
                                  kbe.name, kbe, kbe.score, 1,
                                  False, text_query = True)
            entities.append(ie)
        return entities

    def get_entity_for_mid(self, mid):
        return self.sub_linker.get_entity_for_mid(mid)

    @staticmethod
    def load_stopwords(stopwordsfile):
        stopwords = set()
        with open(stopwordsfile, 'rt', encoding='utf-8') as swfile:
            for word in swfile:
                stopwords.add(word.strip())
        return stopwords


    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = globals.config
        stopwords = EntityLinkerQlever.load_stopwords(
                config_options.get('EntityLinkerQlever',
                'stopwords'))
        sub_linker = EntityLinker.init_from_config()
        backend = sparql_backend.loader.get_backend('qlever')
        return EntityLinkerQlever(sub_linker, backend, stopwords)

    def identify_dates(self, tokens):
        '''
        Identify entities representing dates in the
        tokens.
        :param tokens:
        :return:
        '''
        return self.sub_linker.identify_dates(tokens)

    def identify_entities_in_tokens(self, tokens, min_surface_score=0.1):
        '''
        Identify instances in the tokens.
        :param tokens: A list of string tokens.
        :return: A list of tuples (i, j, e, score) for an identified entity e,
                 at token index i (inclusive) to j (exclusive)
        '''
        entities = self.sub_linker.identify_entities_in_tokens(
                tokens, min_surface_score) if self.sub_linker else []

        logger.info("Trying text queries to get more target entities")
        text_entities = self.textEntityQuery(tokens)
        tfsum = sum([te.score for te in text_entities])
        for te in text_entities:
            logger.info("Text entity "+te.as_string())
            te.surface_score = te.score/tfsum
        entities.extend(text_entities)
        best = sorted(entities, key = lambda x: x.surface_score, reverse = True)
        return best
    

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
