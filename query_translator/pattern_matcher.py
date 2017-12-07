"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from itertools import chain
from . import query_candidate as qc
import logging
from entity_linker.mediator_index_fast import MediatorIndexFast
from entity_linker.entity_linker import Value, DateValue
import time
from . import data
from answer_type.answer_type_identifier import AnswerType
from .alignment import WordembeddingSynonyms, WordDerivations
import config_helper
import math
import sparql_backend.loader

logger = logging.getLogger(__name__)

CONTENT_POS_TAGS = {'NN', 'NNS', 'VB', 'VBD', 'VBN', 'VBZ', 'CD', 'NNP',
                    'JJ', 'VBP', 'JJS', 'RB'}


def get_relation_suffix(relation, suffix_length=3):
    """
    For a relation identifier return the name.
    :param relation:
    :return:
    """
    rel_parts = relation.split('.')
    return '.'.join(rel_parts[-suffix_length:])


def get_relation_name_words(relation):
    """
    For a relation identifier return the name.
    :param relation:
    :return:
    """
    rel_parts = relation.split('.')
    return [c for w in rel_parts for c in w.split('_')]


def get_last_relation_suffix(relation):
    """
    For a relation identifier return the name.
    :param relation:
    :return:
    """
    rel_parts = relation.split('.')
    return rel_parts[-1]


def get_relation_domain(relation):
    """
    For a relation identifier return the name.
    :param relation:
    :return:
    """
    rel_parts = relation.split('.')
    return rel_parts[0]


def get_content_tokens(tokens):
    """
    Remove all tokens that match a whitelist
    of content tokens: nouns, verbs, adjectives
    :param tokens:
    :return:
    """
    content_tokens = [t for t in tokens if t.tag_ in CONTENT_POS_TAGS
                      and t.lemma_ != 'be' and t.lemma_ != 'do'
                      and t.lemma_ != 'go']
    return content_tokens


def filter_relation_suggestions(relation_suggestions):
    """Remove unwanted relation from provided suggestion.
    :return:
    """
    filtered_relations = []
    for relation in relation_suggestions:
        if relation.startswith('http'):
            # Cannot ignore base here because it is needed
            # for dinosaurs and vulcanos.
            continue
        if relation == 'type.object.name':
            # We use type.object.name to get the display name anyway
            # so this relation doesn't add information
            continue
        if '..' in relation:
            # Old frebase dump contain concatenated relations:
            # people.person.spouse_s..people.marriage.from.
            continue
        else:
            filtered_relations.append(relation)
    return filtered_relations


def cosine_similarity(counts_a, counts_b):
    """Compute the cosine similarity.

    :param counts_a:
    :param counts_b:
    :return:
    """
    u = set(chain(counts_a.keys(), counts_b.keys()))
    norm_a = math.sqrt(sum([t ** 2 for t in counts_a.values()]))
    norm_b = math.sqrt(sum([t ** 2 for t in counts_b.values()]))
    norm = norm_a * norm_b
    dot = 0.0
    for k in u:
        if k in counts_a and k in counts_b:
            dot += counts_a[k] * counts_b[k]
    if norm > 0.0:
        dot /= norm
        return dot
    else:
        return 0.0


def kl_divergence(counts_a, counts_b, smooth=True,
                  alpha=1.0):
    """Compute KL-divergence KL(counts_a||counts_b)

    The parameters are maps from class/outcome to count (strictly > 0).
    :param counts_a:
    :param counts_b:
    :return:
    """
    # Laplace smoothing for universe of size d
    total_a = float(sum(counts_a.values()))
    total_b = float(sum(counts_b.values()))
    p_a = dict()
    p_b = dict()
    u = set(chain(counts_a.keys(), counts_b.keys()))
    # Laplace smoothing for both.
    if smooth:
        d = float(len(u))
        for (c, t, r) in zip([counts_a, counts_b],
                             [total_a, total_b],
                             [p_a, p_b]):
            for k in u:
                # Add missing entries.
                if k not in c:
                    r[k] = alpha / (t + alpha * d)
                else:
                    r[k] = (c[k] + alpha) / (t + alpha * d)
    else:
        for (c, t, r) in zip([counts_a, counts_b],
                             [total_a, total_b],
                             [p_a, p_b]):
            for k in c.keys():
                r[k] = c[k] / float(t)
    # Compute the distance.
    kl = 0.0
    for k in u:
        p = p_a[k] * math.log(p_a[k])
        n = p_a[k] * math.log(p_b[k])
        kl += p
        kl -= n
    return kl


def filter_type_distribution(dist, min_count=1,
                             n_max=10):
    """Filter the type distribution: remove base and user types.

    :param dist:
    :return:
    """
    new_dist = dict()
    for key, count in sorted(iter(dist.items()), key=lambda __c1: __c1[1],
                             reverse=True):
        if key.startswith('base') or key.startswith('user'):
            continue
        if dist[key] < min_count:
            continue
        new_dist[key] = dist[key]
        if len(new_dist) == n_max:
            break
    return new_dist


def filter_important_types(relation_target_types, relation_counts,
                           n_min_frac=.1,
                           n_min_total=900):
    """Filter relation target types to the important ones.

    Returns a map with relations as keys and a list of target types
    as value.
    For each relation, a target
    type is only returned if at least a fraction of
    the targets have that type.
    :param relation_target_types:
    :param relation_counts:
    :param n_min_frac:
    :param n_min_total:
    :return:
    """
    logger.info("Filtering target type distributions.")
    important_relation_target_types = dict()
    for relation in relation_target_types.keys():
        if relation in relation_counts:
            # Only allow base classes when the relation starts with base
            total_count = relation_counts[relation][0]
            # Only return the types when at least a fraction of n_min_frac
            # targets has that type.
            important_relation_target_types[relation] = []
            for type_n, count in relation_target_types[relation].items():
                # Ignore base types for non-base relations
                if not relation.startswith('base'):
                    if type_n.startswith('base'):
                        continue
                if count > n_min_total or count / total_count >= n_min_frac:
                    important_relation_target_types[relation].append(type_n)
        else:
            logger.info("No relation count for relation %s." % relation)
            important_relation_target_types[relation] = []
            # Just use the first 10 types as backoff strategy.
            type_names_counts = sorted(relation_target_types[relation].items(),
                                       key=lambda __c: __c[1], reverse=True)
            for type_n, count in type_names_counts[:10]:
                # Ignore base types for non-base relations
                if not relation.startswith('base'):
                    if type_n.startswith('base'):
                        continue
    return important_relation_target_types


class QueryPatternMatcher:
    def __init__(self, query, extender, backend):
        self.extender = extender
        self.query = query
        self.backend = backend

    def construct_initial_query_candidates(self):
        query_candidates = []
        for entity in self.query.identified_entities:
            if isinstance(entity.entity, Value):
                logger.info("Ignoring %s as start entity." % entity.name)
                continue
            query_candidate = qc.QueryCandidate(self.query, self.backend)
            entity_node = qc.QueryCandidateNode(entity.name, entity,
                                                query_candidate)
            query_candidate.root_node = entity_node
            entity_node.set_entity_match(entity)
            query_candidate.set_new_extension(entity_node)
            query_candidates.append(query_candidate)
        return query_candidates

    def match_ERT_pattern(self):
        logger.info("Matching ERT pattern.")
        start = time.time()
        pattern = [self.extender.extend_entity_with_target_relation]
        query_candidates = self.construct_initial_query_candidates()
        query_candidates = self.match_pattern(query_candidates, pattern)
        for query_candidate in query_candidates:
            query_candidate.pattern = "ERT"
        duration = (time.time() - start) * 1000
        logger.info("ERT matched %s times in %.2f ms." % (len(query_candidates),
                                                          duration))
        return query_candidates

    def match_ERMRT_pattern(self):
        logger.info("Matching ERMRT pattern.")
        start = time.time()
        pattern_a = [self.extender.extend_entity_with_mediator,
                     self.extender.extend_mediator_with_targetrelation]
        pattern_b = [
            self.extender.extend_entity_with_mediator_and_targetrelation]
        query_candidates_a = self.construct_initial_query_candidates()
        query_candidates_a = self.match_pattern(query_candidates_a, pattern_a)
        query_candidates_b = self.construct_initial_query_candidates()
        query_candidates_b = self.match_pattern(query_candidates_b, pattern_b)
        query_candidates = query_candidates_a + query_candidates_b
        for query_candidate in query_candidates:
            query_candidate.pattern = "ERMRT"
        duration = (time.time() - start) * 1000
        logger.info(
            "ERMRT matched %s times in %.2f ms." % (len(query_candidates),
                                                    duration))
        return query_candidates

    def match_ERMRERT_pattern(self):
        logger.info("Matching ERMRERT pattern.")
        start = time.time()

        query_candidates = []
        for query_candidate in query_candidates:
            query_candidate.pattern = "ERMRERT"
        # Match the other pattern.
        pattern = [self.extender.extend_entity_with_mediator_and_entity,
                   self.extender.extend_mediator_with_targetrelation]
        other_query_candidates = self.construct_initial_query_candidates()
        other_query_candidates = self.match_pattern(other_query_candidates,
                                                    pattern)
        query_candidates = query_candidates + other_query_candidates
        for query_candidate in query_candidates:
            query_candidate.pattern = "ERMRERT"
        duration = (time.time() - start) * 1000
        logger.info(
            "ERMRERT matched %s times in %.2f ms." % (len(query_candidates),
                                                      duration))
        return query_candidates

    def match_pattern(self, query_candidates, pattern):
        """
        Recursively apply the pattern.
        :param query_candidates:
        :param pattern:
        :return:
        """
        if len(pattern) == 0:
            return query_candidates
        new_candidates = []
        extender = pattern.pop(0)
        # A list means we match the pattern (corresponding to the list)
        # below the current extension. Afterwards we continue at the same
        # extension.
        if isinstance(extender, list):
            # How often did the current pattern set a new extension
            # Note: lists don't advance the current extension
            extender_length = len(
                [x for x in extender if not isinstance(x, list)])
            extended = self.match_pattern(query_candidates, extender)
            # Update the query candidates current extension and extension
            # history.
            for e in extended:
                e.current_extension = e.extension_history[-extender_length]
                e.extension_history = e.extension_history[:-extender_length]
            return self.match_pattern(extended, pattern)
        else:
            for query_candidate in query_candidates:
                extended = extender(query_candidate)
                new_candidates += extended
            return self.match_pattern(new_candidates, pattern)


class QueryCandidateExtender:
    def __init__(self, mediator_index, relation_counts, mediator_names,
                 mediator_relations, reverse_relations, relation_expected_types,
                 backend, relation_words, mediated_rel_words,
                 target_type_distributions, synonyms, word_derivations,
                 word_type_counts, relation_lemmas):
        """
        :type parameters: TranslatorParameters
        :param mediator_index:
        :param relation_counts:
        :param mediator_names:
        :param mediator_relations:
        :param reverse_relations:
        :param relation_expected_types:
        :param backend:
        :param relation_words:
        :param mediated_rel_words:
        :param target_type_distributions:
        :param synonyms:
        :param word_derivations:
        :param word_type_counts:
        :param relation_lemmas:
        :return:
        """
        self.synonyms = synonyms
        self.mediator_index = mediator_index
        self.relation_counts = relation_counts
        # A map from mediator mid to a readable name
        self.mediator_names = mediator_names
        # A list of relations that lead to or come from mediators
        self.mediator_relations = mediator_relations
        # The reverse map: from mediator name to mid
        self.reverse_mediator_names = {self.mediator_names[k]: k
                                       for k in self.mediator_names.keys()}
        self.reverse_relations = reverse_relations
        self.relation_expected_types = relation_expected_types
        self.word_type_counts = word_type_counts
        self.relation_target_type_distributions = target_type_distributions
        self.relation_target_types = filter_important_types(
            target_type_distributions,
            relation_counts)
        self.relation_lemmas = relation_lemmas
        self.backend = backend
        self.relation_words = relation_words
        self.word_derivations = word_derivations
        # A map from rel_a -> [(rel_b, words) ...]
        # where rel_a is the direction <?x rel_a mediator>
        # and rel_b is <mediator rel_b ?y>
        self.mediated_rel_words = dict()
        for (rel_a, rel_b), words in mediated_rel_words.items():
            # Both relations are expressed in the direction with the mediator as subject.
            if rel_a in self.reverse_relations:
                reverse_rel_a = self.reverse_relations[rel_a]
                if reverse_rel_a not in self.mediated_rel_words:
                    self.mediated_rel_words[reverse_rel_a] = []
                self.mediated_rel_words[reverse_rel_a].append((rel_b, words))
            if rel_b in self.reverse_relations:
                reverse_rel_b = self.reverse_relations[rel_b]
                if reverse_rel_b not in self.mediated_rel_words:
                    self.mediated_rel_words[reverse_rel_b] = []
                self.mediated_rel_words[reverse_rel_b].append((rel_a, words))
        # The filter domains used for comparing with GraphParser results
        # self.filter_domains = {'film', 'people', 'business', 'organization',
        #                       'measurement_unit'}
        self.filter_domains = None
        # This needs to be set via set_parameters!
        self.parameters = None

    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = config_helper.config
        backend_module_name = config_options.get("Backend", "backend")
        backend = sparql_backend.loader.get_backend(backend_module_name)
        relation_counts_file = config_options.get('QueryCandidateExtender',
                                                  'relation-counts')
        mediator_names_file = config_options.get('QueryCandidateExtender',
                                                 'mediator-names')
        reverse_relations_file = config_options.get('QueryCandidateExtender',
                                                    'reverse-relations')
        expected_types_file = config_options.get('QueryCandidateExtender',
                                                 'relation-expected-types')
        tt_distributions_file = config_options.get('QueryCandidateExtender',
                                                   'relation-target-type-distributions')
        mediator_relations_file = config_options.get('QueryCandidateExtender',
                                                     'mediator-relations')
        rel_lemmas_file = config_options.get('QueryCandidateExtender',
                                             'relation-lemmas')
        relation_words_file = config_options.get('QueryCandidateExtender',
                                                 'relation-words')
        mediated_relation_words_file = config_options.get(
            'QueryCandidateExtender',
            'mediated-relation-words')
        word_type_counts_file = config_options.get(
            'QueryCandidateExtender',
            'word-type-counts')
        word_type_counts = data.read_word_type_distributions(
            word_type_counts_file)
        embeddings_model = config_options.get('Alignment',
                                              'word-embeddings')
        word_deriv_file = config_options.get('Alignment',
                                             'word-derivations')
        we_synonyms = WordembeddingSynonyms(embeddings_model)
        word_derivations = WordDerivations(word_deriv_file)
        mediator_relations = data.read_mediator_relations(
            mediator_relations_file)
        relation_counts = data.read_relation_counts(relation_counts_file)
        mediator_names = data.read_mediator_names(mediator_names_file)
        mediator_index = MediatorIndexFast.init_from_config()
        reverse_relations = data.read_reverse_relations(reverse_relations_file)
        relation_expected_types = data.read_relation_expected_types(
            expected_types_file)
        relation_words = data.read_relation_words(relation_words_file,
                                                  n_top_words=1000)
        mediated_relation_words = data.read_mediated_relation_words(
            mediated_relation_words_file, n_top_words=1000)
        rel_tt_distributions = data.read_relation_target_type_distributions(
            tt_distributions_file)
        rel_lemmas = data.read_relation_lemmas(rel_lemmas_file)
        return QueryCandidateExtender(mediator_index, relation_counts,
                                      mediator_names,
                                      mediator_relations,
                                      reverse_relations,
                                      relation_expected_types,
                                      backend, relation_words,
                                      mediated_relation_words,
                                      rel_tt_distributions, we_synonyms,
                                      word_derivations, word_type_counts,
                                      rel_lemmas)

    def set_parameters(self, parameters):
        """Sets the parameters of the extender.
        :type parameters TranslatorParameters
        :return:
        """
        self.parameters = parameters

    def get_relation_lemma_name(self, relation):
        if relation in self.relation_lemmas:
            return self.relation_lemmas[relation]
        else:
            logger.info("No lemma version for relation %s." % relation)
            return relation

    def match_relation_with_tokens(self, relation, tokens, query):
        """
        Try to match the relation against the tokens. Returns
        a RelationMatch object.
        :param relation:
        :param token:
        :return:
        """
        # TODO(Elmar): move matching functions to alignment module.
        if query.relation_oracle:
            if query.relation_oracle.is_relation_in_query(query, relation,
                                                          self.reverse_relations):
                return query.tokens
            else:
                return []
        lemma_rel = self.get_relation_lemma_name(relation)
        relation_suffix_words = get_relation_name_words(
            get_relation_suffix(lemma_rel))
        relation_name_words = get_relation_name_words(
            get_relation_suffix(lemma_rel, suffix_length=2))
        relation_words = None
        if relation in self.relation_words:
            relation_words = self.relation_words[relation]
        match = qc.RelationMatch(relation)
        for t in tokens:
            lemmas = [t.lemma_]
            if len(t.lemma_.split('-')) > 1:
                lemmas.extend(t.lemma_.split('-'))
            for l in lemmas:
                # Ignore single word lemmas, e.g. from "nintendo ds":
                # "ds" -> "d".
                if len(l) == 1:
                    continue
                # Match literally in the relation name.
                for w in relation_suffix_words:
                    if w.startswith(l):
                        match.add_relation_name_match(t, w)
                # Match in the relation name via word derivations.
                for w in relation_suffix_words:
                    if self.word_derivations.is_derivation(t, w):
                        match.add_derivation_match(t, w)
                # Match via weak synonyms in the relation name.
                for w in relation_name_words:
                    similarity = self.synonyms.synonym_score(l, w)
                    if similarity > 0:
                        match.add_relation_name_weak_match(t, w, similarity)
                # Match in the computed relation-context words.
                if relation_words and l in relation_words:
                    match.add_relation_words_match(t, relation_words[l])
        return match

    def match_mediated_relation_with_tokens(self, relation_a, relation_b,
                                            tokens, query,
                                            match_relation_name=False):
        """
        Try to match a mediated relation of the form and direction
        <x> <relation_a> <mediator> <relation_b> <y> against the tokens.
        Returns a list of RelationMatch that matched. relation_b can be
        None in which case all matching relations starting with relation_a
        are returned.
        This only considers matching in the relation name if explicitly specified
        via match_relation_name.
        :param relation_a:
        :param token:
        :return:
        """
        if query.relation_oracle:
            result = []
            # Check if rel_a is in the query.
            if query.relation_oracle.is_relation_in_query(query, relation_a,
                                                          self.reverse_relations):
                if relation_a in self.mediated_rel_words:
                    # Check if also rel_b is in the query.
                    relb_words = self.mediated_rel_words[relation_a]
                    for rel_b, rel_words in relb_words:
                        if query.relation_oracle.is_relation_in_query(query,
                                                                      relation_a,
                                                                      self.reverse_relations):
                            result.append((rel_b, query.tokens))
            return result
        result = []
        relb_words = []
        if relation_a in self.mediated_rel_words:
            relb_words = self.mediated_rel_words[relation_a]
        # If relation_b is provided only consider combinations
        # that contain it.
        if relation_b:
            filtered_relb_words = []
            for rel_b, rel_words in relb_words:
                if rel_b != relation_b:
                    continue
                else:
                    filtered_relb_words.append((rel_b, rel_words))
                    break
            if filtered_relb_words:
                relb_words = filtered_relb_words
            # For some combinations we might not have a relation context word
            # so just go with an empty set here.
            else:
                relb_words = [(relation_b, {})]
        for rel_b, rel_words in relb_words:
            match = qc.RelationMatch((relation_a, rel_b))
            rel_a_suffix_words = []
            rel_a_name_words = []
            rel_b_suffix_words = []
            rel_b_name_words = []
            if match_relation_name:
                lemma_rel_a = self.get_relation_lemma_name(relation_a)
                lemma_rel_b = self.get_relation_lemma_name(rel_b)
                rel_a_suffix_words = get_relation_name_words(
                    get_relation_suffix(lemma_rel_a))
                rel_a_name_words = get_relation_name_words(
                    get_relation_suffix(lemma_rel_a, suffix_length=2))
                rel_b_suffix_words = get_relation_name_words(
                    get_relation_suffix(lemma_rel_b))
                rel_b_name_words = get_relation_name_words(
                    get_relation_suffix(lemma_rel_b, suffix_length=2))
            for t in tokens:
                lemma = t.lemma_.replace('-', '_')
                # Ignore single word lemmas, e.g. from "dintendo ds":
                # "ds" -> "d".
                if len(lemma) == 1:
                    continue
                # Match in the computed relation-context words.
                if lemma in rel_words:
                    match.add_relation_words_match(t, rel_words[lemma])
                if match_relation_name:
                    # Match literally in the relation name.
                    for w in rel_a_suffix_words + rel_b_suffix_words:
                        if w.startswith(lemma):
                            match.add_relation_name_match(t, w)
                    # Match via weak synonyms in the relation name.
                    for w in rel_a_name_words + rel_b_name_words:
                        similarity = self.synonyms.synonym_score(lemma, w)
                        if similarity > 0:
                            match.add_relation_name_weak_match(t, w, similarity)
                    # Match in the relation name via word derivations.
                    for w in rel_a_suffix_words + rel_b_suffix_words:
                        if self.word_derivations.is_derivation(t, w):
                            match.add_derivation_match(t, w)

            if not self.parameters.require_relation_match or \
                    not match.is_empty():
                result.append(match)
        return result

    def lookup_mediator_entity_match(self, start_entity, target_entity, query):
        """
        Returns a list of tuples:
        (rel_a, rel_b) where rel_a is the relation from start_entity
        to a mediator object and rel_b is the relation from the mediator
        object to the target_entity. The list is empty when there is
        no match.
        :param start_entity: A Freebase-Easy entity.
        :param target_entity: A Freebase-Easy entity.
        :return:
        """
        relations = set()
        try:
            mediator_list = self.mediator_index.get_freebase_mediators(
                start_entity,
                target_entity)
            for (_, rel_a, rel_b) in mediator_list:
                rev_rel_a = rel_a
                if rel_a in self.reverse_relations:
                    rev_rel_a = self.reverse_relations[rel_a]
                relations.add((rev_rel_a, rel_a, rel_b))
        except KeyError:
            return []
        if query.relation_oracle:
            # Only return the relations that the oracle returns.
            filtered_relations = set()
            oracle = query.relation_oracle
            for rel_a, rel_b in relations:
                if oracle.is_relation_in_query(query, rel_a,
                                               self.reverse_relations) and \
                        oracle.is_relation_in_query(query, rel_b,
                                                    self.reverse_relations):
                    filtered_relations.add((rel_a, rel_b))
            return filtered_relations
        return relations

    def relation_has_date_target(self, relation_name):
        """
        Return True if the expected type of the given relation
        is a date.
        :param relation_name:
        :return:
        """
        if relation_name in self.relation_expected_types:
            if self.relation_expected_types[relation_name] == 'type.datetime':
                return True

        return False

    def relation_has_int_target(self, relation_name):
        """
        Return True if the expected type of the given relation
        is an integer.
        :param relation_name:
        :return:
        """
        if relation_name in self.relation_expected_types:
            if self.relation_expected_types[relation_name] == 'type.int':
                return True
        return False

    def relation_points_to_count(self, relation_name):
        """
        Return True if the target points to a count, e.g.
        for "...number_of.." relations. Currently, this only
        depends on the relation target type to be an integer.
        :param relation_name:
        :return:
        """
        if self.relation_has_int_target(relation_name):
            return True
        return False

    def relation_answers_target_class(self, relation, target_class):
        """Does the relation answer with entities of target_class.

        :param relation:
        :param target_class:
        :return:
        """
        if target_class == 'date' or target_class == 'count':
            # handled separately
            return False
        target_class_suffix = target_class[target_class.rfind('.')+1:]
        # Sometimes the class is meantioned in the relation.
        if target_class_suffix in get_relation_suffix(relation, suffix_length=1):
            return True
        # Check if the class is in the relation type distribution.
        elif relation in self.relation_target_types:
            types = self.relation_target_types[relation]
            for t in types:
                if target_class == t:
                    return True
        # Check if the class is in the expected_type of the relation.
        elif relation in self.relation_expected_types:
            if target_class_suffix in self.relation_expected_types[relation]:
                return True
        return False

    def relation_matches_answer_type(self, relation, query_candidate):
        """Check if the relation matches the answer type.
        :param relation:
        :param query_candidate:
        :return:
        """
        matches_answer_type = 0.0
        for target_class, prob in query_candidate.query.target_type.target_classes:
            matches = False
            if target_class == 'UNK':
                matches = False
            if target_class == 'date' and self.relation_has_date_target(relation):
                matches = True
            if target_class == 'count' and self.relation_points_to_count(relation):
                matches = True
            elif self.relation_answers_target_class(relation, target_class):
                matches = True

            if matches:
                logger.debug("%s matches target_class: %s", relation, target_class)
                matches_answer_type += prob
        return matches_answer_type

    def get_relation_suggestions(self, query_candidate):
        """Return the relation suggestions for the candidate.

        Also, filter them.

        :param query_candidate:
        :return:
        """
        relations = query_candidate.get_relation_suggestions()
        relations = filter_relation_suggestions(relations)
        if self.filter_domains:
            domain_rels = []
            for r in relations:
                domain = get_relation_domain(r)
                if domain in self.filter_domains:
                    domain_rels.append(r)
                elif r in self.reverse_relations:
                    domain = get_relation_domain(self.reverse_relations[r])
                    if domain in self.filter_domains:
                        domain_rels.append(r)
            return domain_rels
        return relations

    def extend_entity_with_target_relation(self, query_candidate):
        query_candidates = []
        query = query_candidate.query
        # Get all the relations for the entity.
        relations = self.get_relation_suggestions(query_candidate)
        remaining_query_content_tokens = get_content_tokens(
            query_candidate.unmatched_tokens)
        # Find the relations that match.
        for rel in relations:

            # Ignore mediators here.
            if rel in self.mediator_relations:
                continue

            # match = self.compute_answer_type_match(rel, query_candidate, )
            # logger.info((rel, match))
            at_match = self.relation_matches_answer_type(rel, query_candidate)
            # Do we require an answer type match?
            if self.parameters.restrict_answer_type and at_match <  0.3:
                continue

            # Try to match remaining token to the relation.
            relation_match = self.match_relation_with_tokens(rel,
                                                             remaining_query_content_tokens,
                                                             query)
            if not self.parameters.require_relation_match or \
                    not relation_match.is_empty():
                if rel in self.relation_counts:
                    cardinality = self.relation_counts[rel]
                    relation_match.cardinality = cardinality

                new_query_candidate = query_candidate.extend_with_relation_and_variable(
                    rel,
                    rel,
                    relation_match)
                new_query_candidate.target_nodes = [
                    new_query_candidate.current_extension]
                new_query_candidate.matches_answer_type = at_match
                if self.relation_points_to_count(rel):
                    new_query_candidate.target_is_count = True
                query_candidates.append(new_query_candidate)
        return query_candidates

    def extend_entity_with_mediator_and_entity(self, query_candidate):
        query_candidates = []
        query = query_candidate.query

        entity = query_candidate.current_extension.entity_match
        identified_entities = query_candidate.query.identified_entities
        start = time.time()
        remaining_query_content_tokens = set(
            get_content_tokens(query_candidate.unmatched_tokens))
        for other_entity in identified_entities:
            # Not allowed to match an entity twice
            if other_entity in query_candidate.matched_entities:
                continue
            # The two entities may not overlap.
            if entity.overlaps(other_entity):
                continue
            relations = self.lookup_mediator_entity_match(entity.sparql_name(),
                                                          other_entity.sparql_name(),
                                                          query)
            # Remove the other entity's tokens for relation matches
            new_query_content_tokens = remaining_query_content_tokens - set(
                other_entity.tokens)
            for rel_a, rev_rel_a, rel_b in relations:
                relation_matches = self.match_mediated_relation_with_tokens(
                    rel_a,
                    rel_b,
                    new_query_content_tokens,
                    query,
                    match_relation_name=True)
                # relation_matches can consists of 0 or 1 matches
                match = None
                if relation_matches:
                    match = relation_matches[0]
                else:
                    match = qc.RelationMatch((rel_a, rel_b))
                new_candidate = query_candidate.extend_with_relation_and_variable(
                    rel_a,
                    rev_rel_a,
                    match,
                    allow_new_match=True)
                new_candidate.extend_with_relation_and_entity(rel_b,
                                                              match,
                                                              other_entity,
                                                              create_copy=False,
                                                              allow_new_match=True)
                new_candidate.current_extension = \
                    new_candidate.extension_history[-2]
                query_candidates.append(new_candidate)
        duration = (time.time() - start) * 1000
        logger.debug("Finding ERMRE pairs took %s ms." % duration)
        return query_candidates

    def extend_entity_with_mediator_and_targetrelation(self, query_candidate):
        query_candidates = []
        query = query_candidate.query

        # Get all the relations for the entity.
        relations = self.get_relation_suggestions(query_candidate)
        remaining_query_content_tokens = get_content_tokens(
            query_candidate.unmatched_tokens)
        # Find the relations that match.
        for rel in relations:
            # Ignore mediators here.
            if rel not in self.mediator_relations:
                continue
            rev_rel = rel
            if rel in self.reverse_relations:
                rev_rel = self.reverse_relations[rel]
            relation_matches = self.match_mediated_relation_with_tokens(rel,
                                                                        None,
                                                                        remaining_query_content_tokens,
                                                                        query,
                                                                        match_relation_name=True)

            for relation_match in relation_matches:
                at_match = self.relation_matches_answer_type(
                    relation_match.relation[1], query_candidate)
                # Check if target relation has correct type.
                if self.parameters.restrict_answer_type and at_match < 0.3:
                    continue
                # Use the target part of the mediated relation to look up cardinality.
                if relation_match.relation[1] in self.relation_counts:
                    cardinality = self.relation_counts[
                        relation_match.relation[1]]
                    relation_match.cardinality = cardinality
                new_query_candidate = query_candidate.extend_with_relation_and_variable(
                    rel,
                    rev_rel,
                    relation_match)
                new_query_candidate.extend_with_relation_and_variable(
                    relation_match.relation[1],
                    relation_match.relation[1],
                    relation_match,
                    create_copy=False)
                new_query_candidate.target_nodes = [
                    new_query_candidate.current_extension]
                new_query_candidate.matches_answer_type = at_match
                if self.relation_points_to_count(relation_match.relation[1]):
                    new_query_candidate.target_is_count = True
                query_candidates.append(new_query_candidate)
        return query_candidates

    def extend_entity_with_mediator(self, query_candidate):
        query_candidates = []
        query = query_candidate.query

        # Get all the relations for the entity.
        relations = self.get_relation_suggestions(query_candidate)
        remaining_query_content_tokens = get_content_tokens(
            query_candidate.unmatched_tokens)
        # Find the relations that match.
        for rel in relations:
            # Only consider mediators here.
            if rel not in self.mediator_relations:
                continue
            rev_rel = rel
            if rel in self.reverse_relations:
                rev_rel = self.reverse_relations[rel]
            relation_match = self.match_relation_with_tokens(rel,
                                                             remaining_query_content_tokens,
                                                             query)
            if not self.parameters.require_relation_match \
                    or not relation_match.is_empty():
                new_query_candidate = query_candidate.extend_with_relation_and_variable(
                    rel,
                    rev_rel,
                    relation_match,
                    allow_new_match=True)
                query_candidates.append(new_query_candidate)
        return query_candidates

    def extend_mediator_with_targetrelation(self, query_candidate):
        query_candidates = []
        query = query_candidate.query
        query_mediator_node = query_candidate.current_extension
        relations = self.get_relation_suggestions(query_candidate)
        # Filled mediator relation slots.
        filled_rev_rel_slots = {self.reverse_relations[r.name] for r in
                                query_mediator_node.in_relations
                                if r.name in self.reverse_relations}
        filled_rel_slots = {r.name for r in query_mediator_node.out_relations}
        filled_relation_slots = filled_rel_slots | filled_rev_rel_slots
        remaining_query_content_tokens = get_content_tokens(
            query_candidate.unmatched_tokens)
        if self.parameters.restrict_answer_type:
            # Only consider unused relations that have correct type.
            relations = [r for r in relations
                         if
                         self.relation_matches_answer_type(r, query_candidate) > 0.3
                         and r not in filled_relation_slots]
        else:
            relations = [r for r in relations
                         if r not in filled_relation_slots]
        # Find possible target relations
        matched_via_token = False
        for relation in relations:
            relation_match = self.match_relation_with_tokens(relation,
                                                             remaining_query_content_tokens,
                                                             query)
            if not self.parameters.require_relation_match \
                    or not relation_match.is_empty():
                if relation in self.relation_counts:
                    cardinality = self.relation_counts[relation]
                    relation_match.cardinality = cardinality
                new_query_candidate = query_candidate.extend_with_relation_and_variable(
                    relation,
                    relation,
                    relation_match,
                    allow_new_match=True)
                new_query_candidate.target_nodes = [
                    new_query_candidate.current_extension]
                # If the target relation points to a count, notify the candidate of that.
                if self.relation_points_to_count(relation):
                    new_query_candidate.target_is_count = True
                at_match = self.relation_matches_answer_type(relation,
                                                             query_candidate)
                new_query_candidate.matches_answer_type = at_match
                query_candidates.append(new_query_candidate)
                matched_via_token = True
        # Only match via counts when no match via tokens was possible.
        if not matched_via_token:
            target_relation_candidates = []
            for relation in relations:
                if relation in self.relation_counts:
                    counts = self.relation_counts[relation]
                    target_relation_candidates.append((relation, counts))
                else:
                    logger.info("Ignoring relation because no "
                                "count available: %s." % relation)
            if target_relation_candidates:
                # Use unique objects as count
                max_rel, max_count = max(target_relation_candidates,
                                         key=lambda x: x[1][2])
                match = qc.RelationMatch(max_rel)
                match.add_count_match(max_count[2])
                match.cardinality = max_count
                new_query_candidate = query_candidate.extend_with_relation_and_variable(
                    max_rel,
                    match,
                    True)
                new_query_candidate.target_nodes = [
                    new_query_candidate.current_extension]
                # If the target relation points to a count, notify the candidate of that.
                if self.relation_points_to_count(max_rel):
                    new_query_candidate.target_is_count = True
                at_match = self.relation_matches_answer_type(max_rel,
                                                             query_candidate)
                new_query_candidate.matches_answer_type = at_match
                query_candidates.append(new_query_candidate)
                query_candidate.target_nodes = [
                    new_query_candidate.current_extension]
        return query_candidates
