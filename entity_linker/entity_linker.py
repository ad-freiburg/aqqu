"""
An approach to identify entities in a query. Uses a custom index for entity information.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import logging
import re
import copy
import time
from entity_linker.util import normalize_entity_name,\
    remove_prefixes_from_name, remove_suffixes_from_name
import config_helper

logger = logging.getLogger(__name__)


class Entity(object):
    """An entity.

    There are different types of entities inheriting from this class, e.g.,
    knowledge base entities and values.
    """

    def __init__(self, name):
        self.name = name

    def sparql_name(self):
        """Returns an id w/o sparql prefix."""
        pass

    def prefixed_sparql_name(self, prefix):
        """Returns an id with sparql prefix."""
        pass


class KBEntity(Entity):
    """A KB entity."""

    def __init__(self, name, identifier, score, aliases):
        Entity.__init__(self, name)
        # The unique identifier used in the knowledge base.
        self.id = identifier
        # A popularity score.
        self.score = score
        # The entity's aliases.
        self.aliases = aliases

    def sparql_name(self):
        return self.id

    def prefixed_sparql_name(self, prefix):
        return "%s:%s" % (prefix, self.id)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Value(Entity):
    """A value.

     Also has a name identical to its value."""

    def __init__(self, name, value):
        Entity.__init__(self, name)
        self.value = value

    def sparql_name(self):
        return self.value

    def prefixed_sparql_name(self, prefix):
        return "%s:%s" % (prefix, self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class DateValue(Value):
    """A date.

    It returns a different sparql name from a value or normal entity.
    """

    def __init__(self, name, date):
        Value.__init__(self, name, date)

    def sparql_name(self):
        return self.value

    def prefixed_sparql_name(self, prefix):
        # Old version uses lowercase t in dateTime
        #return '"%s"^^xsd:dateTime' % self.value
        return '"%s"^^xsd:datetime' % self.value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class IdentifiedEntity:
    """An entity identified in some text."""

    def __init__(self, tokens,
                 name, entity,
                 types,
                 category,
                 score=0, surface_score=0,
                 perfect_match=False,
                 text_match=False,
                 text_query=False,
                 ):
        # A readable name to be displayed to the user.
        self.name = name
        # The tokens that matched this entity.
        self.tokens = tokens
        # A score for the match of those tokens.
        self.surface_score = surface_score
        # A popularity score of the entity.
        self.score = score
        # The identified entity object.
        self.entity = entity
        # A flag indicating whether the entity perfectly
        # matched the tokens.
        self.perfect_match = perfect_match
        # Indicates whether the entity was matched as text and in the question
        self.text_match = text_match
        # A flag indicating if this entity was obtained via text query
        self.text_query = text_query
        # The possible types of this entity in order of relevance descending
        self.types = types
        # The FreebaseEasy category of this entity
        self.category = category

    def as_string(self):
        t = ','.join(["%s" % t.orth_
                      for t in self.tokens])
        return "%s: tokens:%s prob:%.3f score:%s perfect_match:%s text_match:%s text_query:%s" % \
               (self.name, t,
                self.surface_score,
                self.score,
                self.perfect_match,
                self.text_match,
                self.text_query)

    def overlaps(self, other):
        """Check whether the other identified entity overlaps this one."""
        return set(self.tokens) & set(other.tokens)

    def sparql_name(self):
        return self.entity.sparql_name()

    def prefixed_sparql_name(self, prefix):
        return self.entity.prefixed_sparql_name(prefix)

    def __deepcopy__(self, memo):
        # Don't copy tokens, name they are immutable
        res = IdentifiedEntity(
                self.tokens,
                self.name,
                copy.deepcopy(self.entity, memo),
                copy.deepcopy(self.types, memo),
                copy.deepcopy(self.category, memo),
                copy.deepcopy(self.score, memo),
                copy.deepcopy(self.surface_score, memo),
                copy.deepcopy(self.perfect_match, memo),
                copy.deepcopy(self.text_match, memo)
                )
        return res


def get_value_for_year(year):
    """Return the correct value representation for a year."""
    # Older Freebase versions do not have the long suffix.
    #return "%s-01-01T00:00:00+01:00" % (year)
    return "%s" % year


class EntityLinker:

    def __init__(self, entity_index,
                 max_entities_per_tokens=4,
                 max_types=3):
        self.entity_index = entity_index
        self.max_entities_per_tokens = max_entities_per_tokens
        self.max_types = max_types
        # Entities are a mix of nouns, adjectives and numbers and
        # a LOT of other stuff as it turns out:
        # UH, . for: hey arnold!
        # MD for: ben may library
        # PRP for: henry i
        # CD for: episode 1
        # HYPH for "the amazing spider-man"
        # XX for abc (news)
        # FW for "draco _malloy_", "annie"
        # WP for "_niall_ ferguson"
        self.valid_entity_tag = re.compile(r'^(UH|\.|TO|PRP.?|#|FW|IN|VB.?|'
                                           r'RB|CC|HYPH|WP|XX|NNP.?|NN.?|JJ.?|CD|DT|MD|'
                                           r'POS)+$')
        self.ignore_lemmas = {'be', 'of', 'the', 'and', 'or', 'a'}
        self.year_re = re.compile(r'[0-9]{4}')

    @staticmethod
    def init_from_config(ranker_params, entity_index):
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = config_helper.config
        max_entities_per_tokens = int(config_options.get('EntityLinker',
                                                      'max-entites-per-tokens'))
        max_types = int(config_options.get('EntityLinker',
                                                      'max-types-per-entity'))


        return EntityLinker(entity_index,
                            max_entities_per_tokens=max_entities_per_tokens,
                            max_types = max_types)



    def _text_matches_main_name(self, entity, text):

        """
        Check if the entity name is a perfect match on the text.
        :param entity:
        :param text:
        :return:
        """
        text = normalize_entity_name(text)
        text = remove_prefixes_from_name(text)
        name = remove_suffixes_from_name(entity.name)
        name = normalize_entity_name(name)
        name = remove_prefixes_from_name(name)
        if name == text:
            return True
        return False

    def is_entity_occurrence(self, tokens, start, end):
        '''
        Return true if the tokens marked by start and end indices
        are a valid entity occurrence.
        :param tokens:
        :param start:
        :param end:
        :return:
        '''

        token_list = tokens[start:end]
        # Entity mentions cannot be empty
        if len(token_list) < 1:
            return False
        # Concatenate POS-tags
        pos_list = [t.tag_ for t in token_list]
        pos_str = ''.join(pos_list)
        # Check if all tokens are in the ignore list.
        if all([t.lemma_ in self.ignore_lemmas for t in token_list]):
            return False

        # Entity mentions cannot start with an ignored lemma
        if token_list[0].lemma_ in self.ignore_lemmas and \
                token_list[0].lemma_ != 'the':
            return False

        # For length 1 only allows nouns and foreign and unknown word types
        elif len(pos_list) == 1 and (pos_list[0].startswith('N') or
                                     pos_list[0].startswith('J') or
                                     pos_list[0] == 'FW' or
                                     pos_list[0] == 'XX') or \
                (len(pos_list) > 1 and self.valid_entity_tag.match(pos_str)):
            # It is not allowed to split a consecutive NNP
            # if it is a single token.
            if len(pos_list) == 1:
                if pos_list[0].startswith('NNP') and start > 0 \
                        and tokens[start - 1].tag_.startswith('NNP'):
                    return False
                elif pos_list[-1].startswith('NNP') and end < len(tokens) \
                        and tokens[end].tag_.startswith('NNP'):
                    return False
            return True
        return False

    def identify_dates(self, tokens):
        '''
        Identify entities representing dates in the
        tokens.
        :param tokens:
        :return:
        '''
        # Very simplistic for now.
        identified_dates = []
        for i, t in enumerate(tokens):
            if t.tag_ == 'CD':
                # A simple match for years.
                if re.match(self.year_re, t.orth_):
                    year = t.orth_
                    e = DateValue(year, get_value_for_year(year))
                    # TODO(schnelle) the year is currently used in training but
                    # should be more specific tokens[i:i+1] gives us a span so
                    # it's consistent with other entities
                    ie = IdentifiedEntity(tokens[i:i+1], e.name, e,
                                          types=['Date'],
                                          category='Date',
                                          perfect_match=True)
                    identified_dates.append(ie)
        return identified_dates

    def identify_in_tokens(self, tokens, min_surface_score=0.1, lax_mode=False):
        '''
        Actual entity identification function with a special lax mode where we
        are less strict
        '''
        n_tokens = len(tokens)
        identified_entities = []
        for start in range(n_tokens):
            for end in range(start + 1, n_tokens + 1):
                entity_tokens = tokens[start:end]
                if not lax_mode and not self.is_entity_occurrence(tokens,
                                                                  start, end):
                    continue
                entity_str = entity_tokens.text
                logger.debug("Checking if '%s' is an entity.", entity_str)
                entities = self.entity_index.get_entities_for_surface(
                    entity_str)
                logger.debug("Found %r raw entities", len(entities))
                # No suggestions.
                if not entities:
                    continue
                for ent, surface_score in entities:
                    # Ignore entities with low surface score.
                    if surface_score < min_surface_score:
                        continue
                    perfect_match = False
                    # Check if the main name of the entity exactly matches the
                    # text.
                    if self._text_matches_main_name(ent, entity_str):
                        logger.debug("Perfect match: %s", entity_str)
                        perfect_match = True
                    types = self.entity_index.get_types_for_mid(ent.id,
                                                                self.max_types)
                    category = self.entity_index.get_category_for_mid(ent.id)
                    ide = IdentifiedEntity(tokens[start:end],
                                           ent.name, ent,
                                           types=types,
                                           category=category,
                                           score=ent.score,
                                           surface_score=surface_score, 
                                           perfect_match=perfect_match)
                    # self.boost_entity_score(ie)
                    identified_entities.append(ide)
        return identified_entities

    def identify_entities_in_tokens(self, tokens, min_surface_score=0.1):
        '''
        Identify instances in the tokens.
        :param tokens: A list of string tokens.
        :return: A list of IdentifiedEntity
        '''
        logger.info("Starting entity identification.")
        # First find all candidates.
        identified_entities = []
        start_time = time.time()
        identified_entities.extend(self.identify_in_tokens(tokens, min_surface_score))
        if len(identified_entities) == 0:
            # Without any identified entities we would be unable to find anything for
            # the query so retry this time ignoring POS tags
            logger.info("No entities were found, retry in lax mode")
            identified_entities.extend(self.identify_in_tokens(tokens, min_surface_score/2, lax_mode=True))

        identified_entities.extend(self.identify_dates(tokens))

        duration = (time.time() - start_time) * 1000
        identified_entities = self._filter_identical_entities(identified_entities)
        identified_entities = EntityLinker.prune_entities(identified_entities,
                                                          max_threshold=self.max_entities_per_tokens)
        # Sort by quality
        identified_entities = sorted(identified_entities, key=lambda x: (len(x.tokens),
                                                                         x.surface_score),
                                     reverse=True)
        logging.info("Entity identification took %.2f ms. Identified %s entities." % (duration,
                                                                                      len(identified_entities)))
        return identified_entities, None

    def _filter_identical_entities(self, identified_entities):
        '''
        Some entities are identified twice, once with a prefix/suffix
          and once without.
        :param identified_entities:
        :return:
        '''
        entity_map = {}
        filtered_identifications = []
        for e in identified_entities:
            if e.entity not in entity_map:
                entity_map[e.entity] = []
            entity_map[e.entity].append(e)
        for entity, identifications in entity_map.items():
            if len(identifications) > 1:
                # A list of (token_set, score) for each identification.
                token_sets = [(set(i.tokens), i.surface_score)
                              for i in identifications]
                # Remove identification if its tokens
                # are a subset of another identification
                # with higher surface_score
                while identifications:
                    ident = identifications.pop()
                    tokens = set(ident.tokens)
                    score = ident.surface_score
                    if any([tokens.issubset(x) and score < s
                            for (x, s) in token_sets if x != tokens]):
                        continue
                    filtered_identifications.append(ident)
            else:
                filtered_identifications.append(identifications[0])
        return filtered_identifications

    @staticmethod
    def prune_entities(identified_entities, max_threshold=7):
        token_map = {}
        for e in identified_entities:
            tokens = tuple(e.tokens)
            if tokens not in token_map:
                    token_map[tokens] = []
            token_map[tokens].append(e)
        remove_entities = set()
        for tokens, entities in token_map.items():
            if len(entities) > max_threshold:
                sorted_entities = sorted(entities, key=lambda x: x.surface_score, reverse=True)
                # Ignore the entity if it is not in the top candidates, except, when
                # it is a perfect match.
                #for e in sorted_entities[max_threshold:]:
                #    if not e.perfect_match or e.score <= 3:
                #        remove_entities.add(e)
                remove_entities.update(sorted_entities[max_threshold:])
        filtered_entities = [e for e in identified_entities if e not in remove_entities]
        return filtered_entities

    def boost_entity_score(self, entity):
        if entity.perfect_match:
            entity.score *= 60

    @staticmethod
    def create_consistent_identification_sets(identified_entities):
        logger.info("Computing consistent entity identification sets for %s entities." % len(identified_entities))
        # For each identified entity, the ones it overlaps with
        overlapping_sets = []
        for i, e in enumerate(identified_entities):
            overlapping = set()
            for j, other in enumerate(identified_entities):
                if i == j:
                    continue
                if any([t in other.tokens for t in e.tokens]):
                    overlapping.add(j)
            overlapping_sets.append((i, overlapping))
        maximal_sets = []
        logger.info(overlapping_sets)
        EntityLinker.get_maximal_sets(0, set(), overlapping_sets, maximal_sets)
        #logger.info((maximal_sets))
        result = {frozenset(x) for x in maximal_sets}
        consistent_sets = []
        for s in result:
            consistent_set = set()
            for e_index in s:
                consistent_set.add(identified_entities[e_index])
            consistent_sets.append(consistent_set)
        logger.info("Finished computing %s consistent entity identification sets." % len(consistent_sets))
        return consistent_sets

    @staticmethod
    def get_maximal_sets(i, maximal_set, overlapping_sets, maximal_sets):
        #logger.info("i: %s" % i)
        if i == len(overlapping_sets):
            return
        maximal = True
        # Try to extend the maximal set
        for j, (e, overlapping) in enumerate(overlapping_sets[i:]):
            # The two do not overlap.
            if len(overlapping.intersection(maximal_set)) == 0 and not e in maximal_set:
                new_max_set = set(maximal_set)
                new_max_set.add(e)
                EntityLinker.get_maximal_sets(i + 1, new_max_set,
                                               overlapping_sets, maximal_sets)
                maximal = False
        if maximal:
            maximal_sets.append(maximal_set)


if __name__ == '__main__':
    pass
