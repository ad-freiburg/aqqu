"""
Class describing a QueryCandidate as a graph of nodes and edges. The graph
supports circular relationships.
Each question is translated to several QueryCandidates which are scored and
ranked.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import logging
import sys
from .util import *
import copy
import globals
from entity_linker.entity_linker import KBEntity
import weakref

logger = logging.getLogger(__name__)


# Note: might be a place for ABC
class RelationMatch:
    """
    Describes a match of a relation.
    Supports iterative construction by first instantiating
    and then adding tokens that matched in different parts
    with different scores.
    """

    def __init__(self, relation):
        # The tokens that matched the relation
        self.tokens = set()
        # The name of the relation that matched
        # In the case of a mediated relation, this can be a tuple
        # (rel_a, rel_b) for the structure "rel_a -> M -> rel_b"
        self.relation = relation
        # The match objects, lazily instantiated
        # when adding matches.
        self.words_match = None
        self.name_match = None
        self.name_weak_match = None
        self.count_match = None
        self.derivation_match = None
        # For target relations we track the cardinality
        # of a relation. This is only set for target relations.
        self.cardinality = -1

    def is_empty(self):
        """
        Returns False if the match matched any tokens,
        else returns True.
        :return:
        """
        return len(self.tokens) == 0

    def add_count_match(self, count):
        """
        Add a count for the relation.
        :param token:
        :param score:
        :return:
        """
        self.count_match = CountMatch(count)

    def add_relation_words_match(self, token, score):
        """
        Add a token and score that matched the relation words.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.words_match:
            self.words_match = WordsMatch([(token, score)])
        else:
            self.words_match.add_match((token, score))

    def add_relation_name_match(self, token, name):
        """
        Add a token and score that matched the relation words.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.name_match:
            self.name_match = NameMatch([(token, name)])
        else:
            self.name_match.add_match((token, name))

    def add_derivation_match(self, token, name):
        """Add a token and and the name it matched via some derivation.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.derivation_match:
            self.derivation_match = DerivationMatch([(token, name)])
        else:
            self.derivation_match.add_match((token, name))

    def add_relation_name_weak_match(self, token, name, score):
        """
        Add a token and score that matched the relation words.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.name_weak_match:
            self.name_weak_match = NameWeakMatch([(token, name, score)])
        else:
            self.name_weak_match.add_match((token, name, score))

    def __deepcopy__(self, memo):
        # No need to copy the matches. They remain unchanged after
        # RelationMatch was created.
        m = RelationMatch(self.relation)
        m.words_match = self.words_match
        m.name_match = self.name_match
        m.name_weak_match = self.name_weak_match
        return m

    def as_string(self):
        result = []
        if self.name_match:
            result.append(self.name_match.as_string())
        if self.derivation_match:
            result.append(self.derivation_match.as_string())
        if self.words_match:
            result.append(self.words_match.as_string())
        if self.name_weak_match:
            result.append(self.name_weak_match.as_string())
        if self.count_match:
            result.append(self.count_match.as_string())
        indent = "\n  "
        s = indent.join(result)
        relation_name = self.relation
        if isinstance(self.relation, tuple):
            relation_name = ' -> '.join(self.relation)
        return "%s:%s%s" % (relation_name, indent, s)


class WordsMatch:
    """
    Describes a match against a list of tokens.
    It has two lists: a list of matching tokens, and a list
    with a score for each token indicating how well the corresponding
    token matched.
    """

    def __init__(self, token_scores=[]):
        # A list of tuples (word, score)
        self.token_scores = token_scores

    def add_match(self, token_score):
        self.token_scores.append(token_score)

    def as_string(self):
        s = ','.join(["%s:%.4f" % (t.lemma, s)
                      for t, s in self.token_scores])
        return "RelationContext: %s" % s


class DerivationMatch:
    """A match against a derived word.

    """

    def __init__(self, token_names):
        # A list of token, name tuples
        # where token matched name via some derivation.
        self.token_names = token_names

    def add_match(self, token_name):
        self.token_names.append(token_name)

    def as_string(self):
        s = ','.join(["%s=%s" % (t.lemma, n)
                      for t, n in self.token_names])
        return "DerivationMatch: %s" % s


class CountMatch:
    """
    Describes a match using only the relation count of freebase.
    """

    def __init__(self, count):
        # A list of tuples (word, score)
        self.count = count

    def as_string(self):
        return "Count: %s" % self.count


class NameMatch:
    """
    Describes a match of tokens in the relation name.
    """

    def __init__(self, token_names):
        # A list of token, name tuples
        # where token matched name.
        self.token_names = token_names

    def add_match(self, token_name):
        self.token_names.append(token_name)

    def as_string(self):
        s = ','.join(["%s=%s" % (t.lemma, n)
                      for t, n in self.token_names])
        return "RelationName: %s" % s


class NameWeakMatch:
    """
    Describes a match of tokens in the relation name with a weak
    synonym.
    """

    def __init__(self, token_name_scores=[]):
        # A list of tuples (token, name, score)
        # where token matched in name with score.
        self.token_name_scores = token_name_scores

    def add_match(self, token_name_score):
        self.token_name_scores.append(token_name_score)

    def as_string(self):
        s = ','.join(["%s=%s:%.2f" % (t.lemma, n, s)
                      for t, n, s in self.token_name_scores])
        return "RelationNameSynonym: %s" % s


class EntityMatch:
    """
    Describes a match of an entity in the tokens.
    """

    def __init__(self, entity):
        self.entity = entity
        self.score = None

    def __deepcopy__(self, memo):
        # No need to copy the identified entity.
        m = EntityMatch(self.entity)
        m.score = self.score
        return m

    def as_string(self):
        return self.entity.as_string()


class QueryCandidate:
    """
    The contained object pointing to a root node.
    """

    def __init__(self, query, backend, root_node=None):
        # The query we are translating.
        self.query = query
        # The SPARQL backend we used for generating this query candidate.
        # Used to execute the query candidate (once the candidate is matched)
        self.backend = backend
        # A list of nodes of this candidate.
        self.nodes = []
        # A list of relations of this candidate.
        self.relations = []
        # The root node of the candidate.
        self.root_node = root_node
        if root_node:
            self.nodes.append(root_node)
        # A set of EntityMatches.
        self.matched_entities = set()
        # A set of RelationMatches.
        self.matched_relations = set()
        # Sets of matched and unmatched tokens so far.
        self.matched_tokens = set()
        self.unmatched_tokens = set(query.query_tokens)
        # The current point where we are trying to extend the pattern.
        self.current_extension = None
        # A history of extensions, needed to move back.
        self.extension_history = []
        # The pattern that was used for matching.
        self.pattern = None
        # An internal counter used for generating variable names in the SPARQL query.
        self._next_var = 0
        # A list of nodes that identify the "select" variables in a SPARQL query.
        self.target_nodes = None
        # A score computed for this candidate.
        self.rank_score = None
        # If the query candidate already uses an answer relation that represents a count,
        # this should be set to true. This avoids counting results in the SPARQL
        # query when the result relation already represents a count.
        self.target_is_count = False
        # Cache the result count, so that pickled candidates can access it.
        # A value of -1 indicates no result count exists yet.
        self.cached_result_count = -1
        # An indicator whether the candidate matches the answer type
        self.matches_answer_type = None

    def get_relation_names(self):
        return sorted([r.name for r in self.relations])

    def get_entity_names(self):
        return sorted([me.entity.name for me in self.matched_entities])

    def get_entity_scores(self):
        entities = sorted([me.entity for me in self.matched_entities])
        return [e.score for e in entities]

    def __getstate__(self):
        """
        We do this for pickling. Everything requiring a SPARQL backend
        must be set. The SPARQL backend is not available after unpickling.
        :return:
        """
        # Cause the candidate to cache the result count.
        if self.cached_result_count == -1:
            self.get_result_count()
        d = dict(self.__dict__)
        del d['backend']
        del d['extension_history']
        return d

    def __setstate__(self, d):
        """
        We do this for unpickling.
        :param d:
        :return:
        """
        self.__dict__.update(d)
        self.backend = None
        self.extension_history = []

    def get_result_count(self, use_cached_value=True):
        """
        Returns the number of results of the SPARQL
        query when executed against the sparql backend
        for this query candidate.
        :return:
        """
        if use_cached_value and self.cached_result_count > -1:
            return self.cached_result_count
        if self.backend.supports_count:
            sparql_query = self.to_sparql_query(count_query=True)
            query_result = self.backend.query(sparql_query)
            # The query result should have one row with one column which is a
            # number as result or 0 rows
            try:
                if len(query_result) > 0:
                    result = int(query_result[0][0])
                else:
                    result = 0
            except ValueError:
                result = 0
                logger.warn(
                    "Count query returned funky value: %s." % query_result[0][0])
        else:
            sparql_query = self.to_sparql_query(count_query=True)
            query_result = self.backend.query(sparql_query)
            result = len(query_result)

        # For count queries only check if there is a count or not.
        if self.query.is_count_query:
            if result > 1:
                result = 1
        self.cached_result_count = result
        return result

    def get_result(self, include_name=True):
        """
        Returns the results of the SPARQL
        query when executed against the sparql backend
        for this query candidate.
        :return:
        """
        sparql_query = self.to_sparql_query(include_name=include_name)
        query_result = self.backend.query(sparql_query)
        return query_result

    def get_relation_suggestions(self):
        """
        Return relation suggestions the current extension of the query candidate.
        :return:
        """
        # We want suggestions for the latest addition to the candidate.
        query_current_extension = self.current_extension
        p = QueryCandidateVariable(None, name='p')
        o = QueryCandidateVariable(None, name='o')
        query = self._to_extended_sparql_query(query_current_extension,
                                               p, o, [p])
        result = self.backend.query(query)
        relations_list = []
        # Flatten the list.
        if result:
            relations_list = [c for r in result for c in r]
        return relations_list

    def get_next_var(self):
        """
        Get the next free variable index.
        :return:
        """
        var = self._next_var
        self._next_var += 1
        return var

    def extend_with_relation_and_variable(self, relation, relation_match,
                                          allow_new_match=False,
                                          create_copy=True):
        """
        A function to extend the current query candidate with the
        given relation and a variable node.
        :param query_candidate:
        :param relation:
        :param matching_tokens:
        :param allow_new_match:
        :return:
        """
        if create_copy:
            new_query_candidate = copy.deepcopy(self)
        else:
            new_query_candidate = self
        entity_node = new_query_candidate.current_extension
        target_node = QueryCandidateVariable(new_query_candidate)
        relation_node = QueryCandidateRelation(relation,
                                               new_query_candidate,
                                               entity_node,
                                               target_node)
        if relation_match:
            relation_node.set_relation_match(relation_match,
                                             allow_new_match=allow_new_match)
        new_query_candidate.set_new_extension(target_node)
        return new_query_candidate

    def extend_with_relation_and_entity(self, relation,
                                        relation_match,
                                        identified_entity,
                                        allow_new_match=False,
                                        create_copy=True):
        """
        A function to extend the current query candidate with the
        given relation and a entity node node.
        :param query_candidate:
        :param relation:
        :param matching_tokens:
        :param allow_new_match:
        :return:
        """
        if create_copy:
            new_query_candidate = copy.deepcopy(self)
        else:
            new_query_candidate = self
        entity_node = new_query_candidate.current_extension
        target_node = QueryCandidateNode(identified_entity.name,
                                         identified_entity,
                                         new_query_candidate)
        target_node.set_entity_match(identified_entity)
        relation_node = QueryCandidateRelation(relation,
                                               new_query_candidate,
                                               entity_node,
                                               target_node)
        if relation_match:
            relation_node.set_relation_match(relation_match,
                                             allow_new_match=allow_new_match)
        new_query_candidate.set_new_extension(target_node)
        return new_query_candidate

    def set_new_extension(self, candidate_node):
        self.extension_history.append(candidate_node)
        self.current_extension = candidate_node

    def add_relation_match(self, relation_match, allow_new_match=False):
        self.matched_relations.add(relation_match)
        self.matched_tokens.update(relation_match.tokens)
        if not allow_new_match:
            self.unmatched_tokens = self.unmatched_tokens - self.matched_tokens

    def add_entity_match(self, entity_match):
        self.matched_entities.add(entity_match)
        self.matched_tokens.update(entity_match.entity.tokens)
        self.unmatched_tokens = self.unmatched_tokens - set(
            entity_match.entity.tokens)

    def __deepcopy__(self, memo):
        # Create a new empty query candidate
        new_qc = QueryCandidate(self.query, self.backend, None)
        # Copy the root node, and adjust all pointers to the new candidate.
        memo[id(self)] = new_qc
        new_root = copy.deepcopy(self.root_node, memo)
        # Put the new root node in the candidate.
        new_qc.root_node = new_root
        new_qc.nodes.append(new_root)
        # No need to copy query instance.
        new_qc.current_extension = copy.deepcopy(self.current_extension, memo)
        new_qc.extension_history = copy.deepcopy(self.extension_history, memo)
        # Shallow copies are enough her.
        new_qc.matched_tokens = copy.copy(self.matched_tokens)
        new_qc.unmatched_tokens = copy.copy(self.unmatched_tokens)
        new_qc.nodes = copy.deepcopy(self.nodes, memo)
        new_qc.relations = copy.deepcopy(self.relations, memo)
        new_qc.matched_relations = copy.deepcopy(self.matched_relations, memo)
        new_qc.matched_entities = copy.deepcopy(self.matched_entities, memo)
        new_qc.target_nodes = copy.deepcopy(self.target_nodes, memo)
        new_qc.pattern = self.pattern
        new_qc._next_var = self._next_var
        new_qc.matches_answer_type = self.matches_answer_type
        new_qc.target_is_count = self.target_is_count
        new_qc.cached_result_count = self.cached_result_count
        return new_qc

    def graph_as_string(self, indent=2):
        visited = set()
        s = indent * " " + "QueryCandidate [pattern:%s\n" % self.pattern
        s += self.root_node.graph_as_string(visited, indent)
        s += indent * " " + "]"
        return s

    def graph_as_simple_string(self, indent=0):
        visited = set()
        s = indent * " " + "QueryCandidate pattern:%s\n" % self.pattern
        s += self.root_node.graph_as_simple_string(visited, indent + 2)
        return s

    def to_sparql_query(self, distinct=True,
                        include_name=False, filter_target=True,
                        limit=300, count_query=False):
        """
        Returns a SPARQL query corresponding to this graph,
        or None if some error occurred.
        :param distinct adds a distinct qualifier
        :param include_name adds a triple for getting the human readable name
        :param filter_target filter out the target
        :param limit the limit
        :param count_query whether this query is supposed to count results
        :return:
        """
        # A set of nodes we visited.
        visited = set()
        query_prefix = "PREFIX %s: <%s>\n" % (globals.FREEBASE_SPARQL_PREFIX,
                                              globals.FREEBASE_NS_PREFIX)
        sparql_triples = self.root_node.to_sparql_query_triples(visited)
        triples_string = ' .\n '.join(["%s %s %s" % (
        s.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX),
        p.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX),
        o.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX))
                                       for s, p, o in sparql_triples])
        query_vars = []
        if self.target_nodes:
            query_vars = [var.get_sparql_name() for
                          var in self.target_nodes]
        else:
            var_nodes = []
            for s, p, o in sparql_triples:
                if isinstance(s, QueryCandidateVariable):
                    var_nodes.append(s)
                elif isinstance(o, QueryCandidateVariable):
                    var_nodes.append(o)
            if len(var_nodes) > 1:
                logger.warn("More than one variable node in the graph.")
                return None
            elif len(var_nodes) == 0:
                logger.warn("No variable node in the graph.")
                return None
            else:
                query_vars = [var_nodes[0].get_sparql_name()]
        query_vars_str = ' '.join(query_vars)
        # Prepend ?xname query variables and add optional tripel statements
        if include_name:
            query_vars_str_names = ' '.join("%sname" % var for
                                            var in query_vars)
            query_vars_str += ' ' + query_vars_str_names
            name_relation = globals.FREEBASE_NAME_RELATION if not self.backend.lang_in_relations else \
                globals.FREEBASE_NAME_RELATION+".en"
            query_var_triples = ' .\n '.join("%s %s:%s %sname" % (var,
                                                                  globals.FREEBASE_SPARQL_PREFIX,
                                                                  name_relation,
                                                                  var)
                                             for var in query_vars)
            if self.backend.supports_optional:
                triples_string += ' .\n OPTIONAL {' + query_var_triples + '}'
            else:
                triples_string += ' .\n ' + query_var_triples

        distinct_str = 'DISTINCT' if distinct else ''
        # Filter the target so it is not equal to one of the subjects or objects
        # of the query.
        if filter_target:
            node_strs = set()
            filters = []
            for s, p, o in sparql_triples:
                if not isinstance(s, QueryCandidateVariable):
                    node_strs.add(s.get_prefixed_sparql_name(
                        globals.FREEBASE_SPARQL_PREFIX))
                if not isinstance(o, QueryCandidateVariable):
                    node_strs.add(o.get_prefixed_sparql_name(
                        globals.FREEBASE_SPARQL_PREFIX))
            for var in query_vars:
                for node_str in node_strs:
                    filters.append('%s != %s' % (var, node_str))
            triples_string += ' .\n FILTER (%s)' % (' && '.join(filters))

        # If the query asks for a query count and the target relation is not already
        # a count -> count the results. (Also do this if exlicitly requested).
        if self.backend and self.backend.supports_count and \
                ((self.query.is_count_query and not self.target_is_count) or count_query):
            # For count queries we increase the limit.
            limit *= 100
            # Note: this does not add "distinct" to the inner clause,
            # which is wrong but what sempre did as well.
            query = "%s\nSELECT count(%s) WHERE " \
                    "{\nSELECT %s WHERE {\n %s \n} LIMIT %s" \
                    "\n}" % (query_prefix,
                           query_vars[0],
                           query_vars_str,
                           triples_string, limit)
        # Just prepend the prefix
        else:
            query = "%s\nSELECT %s %s WHERE {\n %s \n} LIMIT %s" % (query_prefix,
                                                                   distinct_str,
                                                                   query_vars_str,
                                                                   triples_string,
                                                                   limit)
        return query

    def _to_extended_sparql_query(self, subject, predicate, object,
                                  query_vars, distinct=True, limit=300):
        """
        Creates a query for the current candidate that is extended by
        subject predicate and object. This is only used to get relation
        suggestions.
        :param subject:
        :param predicate:
        :param object:
        :param query_vars:
        :param distinct:
        :param limit:
        :return:
        """
        # A set of nodes we visited.
        visited = set()
        query_prefix = "PREFIX %s: <%s>\n" % (globals.FREEBASE_SPARQL_PREFIX,
                                              globals.FREEBASE_NS_PREFIX)
        sparql_triples = self.root_node.to_sparql_query_triples(visited)
        triples_string = ' .\n '.join(["%s %s %s" % (
        s.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX),
        p.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX),
        o.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX))
                                       for s, p, o in sparql_triples])
        extension_triple = "%s %s %s" % (
        subject.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX),
        predicate.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX),
        object.get_prefixed_sparql_name(globals.FREEBASE_SPARQL_PREFIX))
        if triples_string:
            triples_string += " .\n " + extension_triple
        else:
            triples_string = extension_triple
        query_vars = ' '.join("%s" % var.get_sparql_name() for
                              var in query_vars)
        distinct_str = 'DISTINCT' if distinct else ''
        query = "%s SELECT %s %s WHERE {\n %s \n} LIMIT %s" % (query_prefix,
                                                               distinct_str,
                                                               query_vars,
                                                               triples_string,
                                                               limit)
        return query

    def _collect_elements(self):
        """Recursively collect all nodes and edges"""
        elements = set()
        self.root_node._collect_elements(elements)
        return elements

    def covered_tokens(self):
        """
        Return the set of tokens covered by this candidate.
                :return
        """
        return self.matched_tokens

    def default_quality_tuple(self):
        """
        Return a tuple representing the 'goodness' of the
        candidate.
        :return:
        """
        elements = self._collect_elements()
        match_score = 0
        for e in elements:
            if e.entity_match is not None:
                match_score += e.entity_match.entity.score
        return len(self.covered_tokens()), match_score


class QueryCandidateNode:
    """
    This is an entity/mediator.
    """

    def __init__(self, name, entity, query_candidate):
        self.entity = entity
        self.query_candidate = None
        # Temporary nodes provide None here so we don't add them.
        if query_candidate:
            self.query_candidate = query_candidate
            query_candidate.nodes.append(self)
        self.relation_match = None
        self.entity_match = None
        self.name = name
        self.out_relations = []
        self.in_relations = []
        self.score = None

    def get_sparql_name(self):
        return self.entity.sparql_name()

    def get_prefixed_sparql_name(self, prefix):
        return self.entity.prefixed_sparql_name(prefix)

    def set_relation_match(self, relation_match, allow_new_match=False):
        self.relation_match = relation_match
        self.query_candidate.add_relation_match(self.relation_match,
                                                allow_new_match=allow_new_match)

    def set_entity_match(self, identified_entity):
        self.entity_match = EntityMatch(identified_entity)
        self.query_candidate.add_entity_match(self.entity_match)

    def as_string(self):
        s = "Entity [name:%s" % self.name
        if self.entity_match:
            s += ", %s" % self.entity_match.as_string()
        if self.relation_match:
            s += ", %s" % self.relation_match.as_string()
        s += "]"
        return s

    def as_simple_string(self):
        if isinstance(self.entity.entity, KBEntity):
            s = "[%s (%s)]" % (self.name, self.entity.entity.id)
        else:
            s = "[%s]" % self.name
        return s

    def graph_as_string(self, visited, indent):
        s = indent * " " + self.as_string() + "\n"
        for r in self.out_relations:
            s += indent * " " + "  ->" + r.as_string() + "\n"
        for r in self.in_relations:
            if not r.has_source():
                s += indent * " " + "  <-" + r.as_string() + "\n"
        visited.add(self)
        for r in self.out_relations:
            if r.has_target() and not r.target_node in visited:
                s += r.target_node.graph_as_string(visited, indent)
        for r in self.in_relations:
            if r.has_source() and not r.source_node in visited:
                s += r.source_node.graph_as_string(visited, indent)
        return s

    def graph_as_simple_string(self, visited, indent):
        if self in visited:
            return ""
        visited.add(self)
        s = ""
        for r in self.out_relations:
            if r.has_target():
                s += indent * " " + "%s -> %s -> %s\n" % (
                self.as_simple_string(),
                r.as_simple_string(),
                r.target_node.as_simple_string())
                if r.target_node not in visited:
                    s += r.target_node.graph_as_simple_string(visited, indent)
            else:
                s += indent * " " + "%s -> %s -> X\n" % (
                self.as_simple_string(),
                r.as_simple_string())
        for r in self.in_relations:
            if not r.has_source():
                s += indent * " " + "%s <- %s <- X\n" % (
                self.as_simple_string(),
                r.as_simple_string())
            elif r.source_node not in visited:
                s += r.source_node.graph_as_simple_string(visited, indent)
        return s

    def to_sparql_query_triples(self, visited):
        # Sanity check.
        if self in visited:
            logger.error('Loop in creating SPARQL query.')
            return ""
        triples = []
        for r in self.out_relations:
            if not r.has_target:
                logger.warn(
                    "Relation %s in the query graph misses a target node." % r.name)
                continue
            triples.append((self, r, r.target_node))
        visited.add(self)
        # Recursively create query:
        for r in self.out_relations:
            if r.has_target() and r.target_node not in visited:
                node = r.target_node
                other_triples = node.to_sparql_query_triples(visited)
                triples += other_triples
        for r in self.in_relations:
            if r.has_source() and not r.source_node in visited:
                node = r.source_node
                other_triples = node.to_sparql_query_triples(visited)
                triples += other_triples
        return triples

    def _collect_elements(self, elements):
        if self in elements:
            return
        else:
            elements.add(self)
        for r in self.out_relations + self.in_relations:
            elements.add(r)
            if r.has_source():
                r.source_node._collect_elements(elements)
            if r.has_target():
                r.target_node._collect_elements(elements)


class QueryCandidateVariable(QueryCandidateNode):
    """
    This represents a variable in the query graph, i.e.
    an unidentified node.
    """

    def __init__(self, query_candidate, name=None):
        """
        Note: for suggestions we often extend the query with
        temporary variables. This consumes numbers, so that the
        final query variables are non-consecutive, which looks
        funky. So we allow to provide a custom name, for the
        temporary queries - make sure they don't clash with existing
        vars, e.g. by using characters.
        :param query_candidate:
        :param name:
        :return:
        """
        if not name:
            self.variable_name = query_candidate.get_next_var()
        else:
            self.variable_name = name
        node_name = "var_%s" % self.variable_name
        QueryCandidateNode.__init__(self, node_name, node_name,
                                    query_candidate)
        # query_candidate.nodes.append(self)

    def get_sparql_name(self):
        return '?%s' % self.variable_name

    def get_prefixed_sparql_name(self, prefix):
        return '?%s' % self.variable_name

    def as_string(self):
        s = "Variable [index:%s" % self.variable_name
        s += "]"
        return s

    def as_simple_string(self):
        s = "[?%s]" % self.variable_name
        return s


class QueryCandidateRelation(QueryCandidateNode):
    """
    This is a relation. But note that a relation
    can also occur in the place of a subject or object, so
    it is a special kind of Node.
    """

    def __init__(self, name, query_candidate, source_node, target_node):
        QueryCandidateNode.__init__(self, name, name, query_candidate)
        self.relation_match = None
        self.entity_match = None
        self.name = name
        self.source_node = source_node
        self.target_node = target_node
        self.reversed = False
        self.score = None
        self.query_candidate.relations.append(self)

        if not self.source_node is None:
            source_node.out_relations.append(self)
        if not self.target_node is None:
            target_node.in_relations.append(self)

    def has_target(self):
        return not self.target_node is None

    def has_source(self):
        return not self.source_node is None

    def as_string(self):
        s = "Relation [name:%s" % self.name
        if self.entity_match:
            s += ", %s" % self.entity_match.as_string()
        if self.relation_match:
            s += ", %s" % self.relation_match.as_string()
        s += "]"
        return s

    def as_simple_string(self):
        s = "[%s]" % self.name
        return s

    def get_sparql_name(self):
        return '%s' % self.name

    def get_prefixed_sparql_name(self, prefix):
        return '%s:%s' % (prefix, self.name)


def test():
    query = Query("some test query")
    query.query_tokens = [1, 2, 3]
    c = QueryCandidate(query)
    root = QueryCandidateNode(':e:Albert_Einstein', 'm.123', c)
    af = QueryCandidateNode(':e:Albert_Flintstein', 'm.234', c)
    ah = QueryCandidateNode(':e:Albert_Hintstein', 'm.345', c)
    av = QueryCandidateVariable(c)
    c.root_node = root
    age = QueryCandidateRelation('people/person/age', c, source_node=root,
                                 target_node=av)
    ae_af = QueryCandidateRelation('ae-af', c, source_node=root, target_node=af)
    af_ah = QueryCandidateRelation('af-ah', c, source_node=af, target_node=ah)
    ah_ae = QueryCandidateRelation('ah-ae', c, source_node=ah, target_node=root)
    print(c.to_sparql_query())
    x = QueryCandidate(query)
    root = QueryCandidateNode(':e:Albert_Einstein', 'm.123', x)
    x.root_node = root
    p = QueryCandidateVariable(x)
    o = QueryCandidateVariable(x)
    print(x._to_extended_sparql_query(root, p, o, [p]))


if __name__ == '__main__':
    from .translator import Query

    test()
