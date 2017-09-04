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
import config_helper
import spacy
import sparql_backend.loader
from .entity_linker import EntityLinker, KBEntity, IdentifiedEntity

logger = logging.getLogger(__name__)

class EntityLinkerQlever(EntityLinker):

    def __init__(self, entity_index, qlever_backend, stopwords,
            max_entities_per_tokens=4, max_text_entities=5):
        super().__init__(entity_index, max_entities_per_tokens)
        self.qlever_backend = qlever_backend
        self.stopwords = stopwords
        self.max_text_entities = max_text_entities
        self.min_subrange = 3

    @staticmethod
    def init_from_config(ranker_params, entity_index):
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = config_helper.config
        stopwords = EntityLinkerQlever.load_stopwords(
                config_options.get('EntityLinkerQlever',
                                   'stopwords'))
        max_es_per_tokens = int(
            config_options.get('EntityLinker',
                               'max-entites-per-tokens'))
        qlever_backend = sparql_backend.loader.get_backend('qlever')
        return EntityLinkerQlever(entity_index, qlever_backend, stopwords,
                                  max_entities_per_tokens=max_es_per_tokens)

    def text_entity_query(self, tokens, limit):
        toks_nostop = [t for t in tokens 
                       if t.lower_ not in self.stopwords]
        entities = []
        max_start = min(len(toks_nostop), 
                        max(0, len(toks_nostop)-self.min_subrange))
        for start in range(max_start+1):
            min_subrange_end = min(start + self.min_subrange, len(toks_nostop))
            for end in range(min_subrange_end, len(toks_nostop)+1):
                new_entities = self.simple_text_entity_query(
                    toks_nostop[start:end], limit)
                entities.extend(new_entities)
        return entities


    def simple_text_entity_query(self, tokens, limit):
        text = ' '.join([t.lower_ for t in tokens])
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
            identified = IdentifiedEntity([],
                                          kbe.name, kbe, kbe.score, 1,
                                          False, text_query=True)
            entities.append(identified)
        return entities

    @staticmethod
    def load_stopwords(stopwordsfile):
        '''
        Load the stopwords file used for filtiering uninteresting
        tokens from the question
        '''
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

        text_entities = self.text_entity_query(tokens, self.max_text_entities)
        text_entity_map = {te.entity.id: te for te in text_entities}
        for entity in entities:
            if hasattr(entity.entity, 'id') and \
                    entity.entity.id in text_entity_map:
                entity.score += text_entity_map[entity.entity.id].score
                entity.surface_score *= 10
                entity.text_match = True

        return entities


