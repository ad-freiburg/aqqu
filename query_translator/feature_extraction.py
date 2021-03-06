"""
A module for extracting features from a query candidate.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""

from collections import defaultdict
import math
import logging
from itertools import chain
from freebase import get_mid_from_qualified_string


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

N_GRAM_STOPWORDS = {'be', 'do', '?', 'the', 'of', 'is', 'are', 'in', 'was',
                    'did', 'does', 'a', 'for', 'have', 'there', 'on', 'has',
                    'to', 'by', 's', 'some', 'were', 'at', 'been', 'do',
                    'and', 'an', 'as'}


def get_ngrams(tokens, n=2):
    """Return n-grams for the given text tokens.

    n-grams are "_"-concatenated tokens.
    :param n:
    :return:
    """
    grams = zip(*[tokens[i:] for i in range(n)])
    return grams


def get_ngram_features(candidate, use_type_names=True):
    """Get ngram features from the query of the candidate.

    :type candidate: QueryCandidate
    :param candidate:
    :param use_type_names - whether to use [entity] or [classname] as entity
                            replacement
    :return:
    """
    query_text_tokens = [x.lower()
                         for x in
                         get_query_text_tokens(candidate,
                                               use_type_names)]
    # First get bi-grams.
    ngrams = get_ngrams(query_text_tokens, n=2)
    # Then get uni-grams.
    return chain(ngrams, get_ngrams(query_text_tokens, n=1))


def get_query_text_tokens(candidate, use_type_names=True):
    """
    Return the query text for the candidate.
    :param candidate:
    :return:
    """
    # The set of all tokens for which an entity was identified.
    entity_tokens = dict()
    for mention in candidate.matched_entities:
        for tok in mention.tokens:
            entity_tokens[tok] = mention
    query_text_tokens = ['<start>']
    # Replace entity tokens with "[<entity_class>]" or "[entity]" if
    # use_type_names==False
    for tok in candidate.query.tokens:
        # ignore punctuation
        if tok.pos_ == 'PUNCT':
            continue
        if tok in entity_tokens:
            entity = entity_tokens[tok]
            if use_type_names:
                placeholder = '['+entity.category.lower().replace(' ', '_')+']'
            else:
                placeholder = '[entity]'
            # Don't replace if the previous token is an entity token.
            # This conflates multiple tokens for the same entity
            # but also multiple entities
            if query_text_tokens and query_text_tokens[-1] == placeholder:
                # only need one per mention
                continue
            query_text_tokens.append(placeholder)
        else:
            query_text_tokens.append(tok.orth_.lower())
    return query_text_tokens

def pattern_complexity(candidate):
    """
    Determines the complexity of a pattern for which
    we use the number of relations in the pattern.
    """
    return candidate.pattern.count('R')


def simple_features(candidate):
    """Extract features from the a single candidate.

    :type candidate: QueryCandidate
    :param candidate:
    :return:
    """
    # The number of literal entities.
    n_literal_entities = 0
    # The number of entities matched in the question and matched in a text query
    n_text_and_question_entities = 0
    # The sum of surface_score * mention_length over all entity mentions.
    em_token_score = 0.0
    # The number of relations that are matched literally at least once.
    n_literal_relations = 0
    # The number of relations that are matched by word at least once.
    n_word_relations = 0
    # The number of relations that are matched by word weakly at least once.
    n_weak_relations = 0
    # The number of relations that are matched by derivative at least once.
    n_derivative_relations = 0
    # The number of tokens that are part of a literal entity match.
    n_literal_entity_tokens = 0
    # The numper of tokens that match literally in a relation.
    n_literal_relation_tokens = 0
    # The number of tokens that match via derivation in a relation.
    n_derivation_relation_tokens = 0
    # The sum of all weak match scores.
    sum_weak_relation_tokens = 0
    # The sum of all weak match scores.
    sum_context_relation_tokens = 0
    # The size of the result.
    result_size = candidate.get_result_count()
    cardinality = 0
    # Each entity match represents a matched entity.
    n_entity_matches = len(candidate.matched_entities)
    em_surface_scores = []
    em_pop_scores = []
    n_entity_tokens = 0
    for em in candidate.matched_entities:
        # A threshold above which we consider the match a literal match.
        threshold = 0.8
        n_entity_tokens += len(em.tokens)
        if em.perfect_match or em.surface_score > threshold:
            n_literal_entities += 1
            n_literal_entity_tokens += len(em.tokens)
        if em.text_match:
            n_text_and_question_entities += 1
        em_surface_scores.append(em.surface_score)
        em_score = em.surface_score
        em_score *= len(em.tokens.text)
        em_token_score += em_score
        if em.score > 0:
            em_pop_scores.append(math.log(em.score))
        else:
            em_pop_scores.append(0)
    token_name_match_score = defaultdict(float)
    token_weak_match_score = defaultdict(float)
    token_word_match_score = defaultdict(float)
    token_derivation_match_score = defaultdict(float)
    for rm in candidate.matched_relations:
        if rm.name_match:
            for (t, _) in rm.name_match.token_names:
                token_name_match_score[t] += 1.0
            n_literal_relations += 1
        if rm.words_match:
            for (t, s) in rm.words_match.token_scores:
                token_word_match_score[t] += s
            n_word_relations += 1
        if rm.name_weak_match:
            for (t, _, s) in rm.name_weak_match.token_name_scores:
                token_weak_match_score[t] += s
            n_weak_relations += 1
        if rm.derivation_match:
            for (t, _) in rm.derivation_match.token_names:
                token_derivation_match_score[t] += 1.0
                n_derivation_relation_tokens += 1
            n_derivative_relations += 1
        # cardinality is only set for the answer relation.
        if rm.cardinality != -1: # this is a tuple but gets initalized as -1
            # Number of facts in the relation (like in FreebaseEasy).
            cardinality = rm.cardinality[0]

    n_literal_relation_tokens = len(token_name_match_score)
    n_word_relation_tokens = len(token_word_match_score)
    n_weak_relation_tokens = len(token_weak_match_score)
    sum_weak_relation_tokens = round(sum(token_weak_match_score.values()), 2)
    sum_context_relation_tokens = round(
        sum(token_word_match_score.values()), 6)
    avg_em_surface_score = round(
        sum(em_surface_scores) / len(em_surface_scores), 2)
    sum_em_surface_score = round(
        sum(em_surface_scores), 2)
    avg_em_popularity = round(sum(em_pop_scores) / len(em_pop_scores), 2)
    sum_em_popularity = round(sum(em_pop_scores), 2)
    cardinality = int(math.log(cardinality)) if cardinality > 0 \
        else cardinality

    # Each of these maps from a token to a relation matching score.
    # We are interested in the set of all tokens.
    token_matches = [token_derivation_match_score,
                     token_weak_match_score,
                     token_name_match_score,
                     token_word_match_score]
    n_rel_tokens = len(set.union(*[set(x.keys()) for x in token_matches]))

    coverage = ((n_rel_tokens + n_entity_tokens) /
                len(candidate.query.tokens))
    features = {}
    n_relations = len(candidate.get_relation_names())
    n_unmatched_relations = n_relations - len(candidate.matched_relations)
    #relation_match = 1 if candidate.matched_relations else 0
    result_size_0 = 1 if result_size == 0 else 0
    result_size_1_to_20 = 1 if result_size > 0 and result_size < 20 else 0
    result_size_gte_20 = 1 if result_size >= 20 else 0
    matches_answer_type = round(candidate.matches_answer_type, 2)

    # Text query features are only used when a text query was run
    # during entity identification. They require fetching the entire result
    text_answer_ratio = 0.0
    if candidate.query.text_entities:
        # fetch the results, this is expensive but
        # at least for those candidates that are not pruned
        # we need to do it anyway and it's not done again
        candidate.retrieve_result()
        text_entity_map = {te.entity.id: te
                           for te in candidate.query.text_entities}
        for row in candidate.query_result:
            result_mid = get_mid_from_qualified_string(row[0])
            if result_mid in text_entity_map:
                text_answer_ratio += 1.0
        text_answer_ratio = text_answer_ratio / len(candidate.query_result) \
            if candidate.query_result else 0.0

    features.update({
        # "General Features
        'pattern_complexity': pattern_complexity(candidate),
        'coverage': coverage,
        'matches_answer_type': matches_answer_type,
        'result_size_0': result_size_0,
        'result_size_1_to_20': result_size_1_to_20,
        'result_size_gte_20': result_size_gte_20,
        'n_total_literal_tokens': (n_literal_entity_tokens
                                   + n_literal_relation_tokens),
    })
    #features.update({
    #    # text features
    #    'n_text_and_question_entities': n_text_and_question_entities,
    #    'text_answer_ratio': text_answer_ratio,
    #})
    features.update({
        # "Entity Features"
        'n_literal_entities': n_literal_entities,
        'n_entity_matches': n_entity_matches,
        'n_literal_entity_tokens': n_literal_entity_tokens,
        'avg_em_surface_score': avg_em_surface_score,
        'sum_em_surface_score': sum_em_surface_score,
        'avg_em_popularity': avg_em_popularity,
        'sum_em_popularity': sum_em_popularity,
    })
    features.update({
        # "Relation Features"
        'n_relations': n_relations,
        #'n_unmatched_relations': n_unmatched_relations,
        'n_literal_relations': n_literal_relations,
        'n_literal_relation_tokens': n_literal_relation_tokens,
        #'n_word_relations': n_word_relations,
        'n_word_relation_tokens': n_word_relation_tokens,
        #'n_weak_relations': n_word_relations,
        'n_weak_relation_tokens': n_weak_relation_tokens,
        #'n_derivative_relations': n_derivative_relations,
        'n_derivation_relation_tokens': n_derivation_relation_tokens,
        'sum_weak_relation_tokens': sum_weak_relation_tokens,
        'sum_context_relation_tokens': sum_context_relation_tokens,
        'cardinality': cardinality,
    })
    return features


def ngram_features(candidate, ngram_dict, use_type_names=True):
    """Extract ngram features from the single candidate.

    :param candidate:
    :param use_type_names - whether to use [entity] or [classname] as entity
                            replacement
    :return:
    """
    ngram_features = dict()

    def add_feature(f_name):
        if ngram_dict is None or f_name in ngram_dict:
            ngram_features[f_name] = 1

    relations = sorted(candidate.get_relation_names())
    all_rels = '_'.join(relations)
    n_grams = get_ngram_features(candidate, use_type_names)
    for ng in n_grams:
        # Ignore ngrams that only consist of stopfwords.
        if set(ng).issubset(N_GRAM_STOPWORDS):
            continue
        f_name = 'rel:%s+word:%s' % (all_rels, '_'.join(ng))
        add_feature(f_name)
    return ngram_features


def extract_ngram_features(candidates, ngram_dict=None):
    features = []
    for c in candidates:
        features.append(ngram_features(c, ngram_dict))
    return features


def extract_features(candidates,
                     rel_score_model=None,
                     deep_rel_score_model=None):
    """Extract features for multiple candidates at once.

    :param candidates:
    :return:
    """
    all_features = []
    for c in candidates:
        feature_dict = simple_features(c)
        all_features.append(feature_dict)

    if deep_rel_score_model:
        deep_rel_scores = deep_rel_score_model.score_multiple(candidates)
        for i, f in enumerate(all_features):
            deep_rel_score_raw = float(deep_rel_scores[i].score)
            f['deep_relation_score'] = max(0.0, deep_rel_score_raw)
    if rel_score_model:
        rel_scores = rel_score_model.score_multiple(candidates)
        for i, f in enumerate(all_features):
            f['relation_score'] = float(rel_scores[i].score)
    return all_features
