"""
A module for extracting features from a query candidate.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""

from .query_candidate import QueryCandidate
from collections import defaultdict
import math
from itertools import chain
from entity_linker.entity_linker import KBEntity

N_GRAM_STOPWORDS = {'be', 'do', '?', 'the', 'of', 'is', 'are', 'in', 'was',
                    'did', 'does', 'a', 'for', 'have', 'there', 'on', 'has',
                    'to', 'by', 's', 'some', 'were', 'at', 'been', 'do',
                    'and', 'an', 'as'}


def get_n_grams(tokens, n=2):
    """Return n-grams for the given text tokens.

    n-grams are "_"-concatenated tokens.
    :param n:
    :return:
    """
    grams = zip(*[tokens[i:] for i in range(n)])
    return grams


def get_n_grams_features(candidate):
    """Get ngram features from the query of the candidate.

    :type candidate: QueryCandidate
    :param candidate:
    :return:
    """
    query_text_tokens = [x.lower() for x in get_query_text_tokens(candidate)]
    # First get bi-grams.
    n_grams = get_n_grams(query_text_tokens, n=2)
    # Then get uni-grams.
    return chain(n_grams, get_n_grams(query_text_tokens, n=1))


def get_query_text_tokens(candidate, include_mid=False):
    """
    Return the query text for the candidate.
    :param candidate:
    :return:
    """
    # The set of all tokens for which an entity was identified.
    entity_tokens = dict()
    for em in candidate.matched_entities:
        for t in em.entity.tokens:
            entity_tokens[t] = em
    query_text_tokens = ['<start>']
    # Replace entity tokens with "ENTITY"
    for t in candidate.query.query_tokens:
        if t in entity_tokens:
            if include_mid and isinstance(entity_tokens[t].entity.entity, KBEntity):
                mid = entity_tokens[t].entity.entity.id
                # Don't repeat the same mid.
                if len(query_text_tokens) > 0 and query_text_tokens[-1] == mid:
                    continue
                query_text_tokens.append(mid)
            else:
                # Don't replace if the previous token is an entity token.
                # This conflates multiple tokens for the same entity
                # but also multiple entities
                if len(query_text_tokens) > 0 and query_text_tokens[-1] == '<entity>':
                    continue
                else:
                    query_text_tokens.append('<entity>')
        else:
            query_text_tokens.append(t.orth_)
    return query_text_tokens


def simple_features(candidate,
                    generic_features=True,
                    entity_features=True):
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
    # A flag whether the candidate contains a mediator.
    is_mediator = 0.0
    # The number of relations that are matched literally at least once.
    n_literal_relations = 0
    # The number of relations that are matched by word at least once.
    n_word_relations = 0
    # The number of relations that are matched by word weakly at least once.
    n_word_weak_relations = 0
    # The number of relations that are matched by derivative at least once.
    n_derivative_relations = 0
    # The number of tokens that are part of a literal entity match.
    literal_entities_length = 0
    # The number of tokens that match literal in a relation.
    n_literal_relation_tokens = 0
    # The number of tokens that match via weak synoynms in a relation.
    n_weak_relation_tokens = 0
    # The number of tokens that match via derivation in a relation.
    n_derivation_relation_tokens = 0
    # The number of tokens that match via relation context in a relation.
    n_context_relation_tokens = 0
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
        n_entity_tokens += len(em.entity.tokens)
        if em.entity.perfect_match or em.entity.surface_score > threshold:
            n_literal_entities += 1
            literal_entities_length += len(em.entity.tokens)
        if em.entity.text_match:
            n_text_and_question_entities += 1
        em_surface_scores.append(em.entity.surface_score)
        em_score = em.entity.surface_score
        em_score *= len(em.entity.tokens)
        em_token_score += em_score
        if em.entity.score > 0:
            em_pop_scores.append(math.log(em.entity.score))
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
            n_word_weak_relations += 1
        if rm.derivation_match:
            for (t, _) in rm.derivation_match.token_names:
                token_derivation_match_score[t] += 1.0
            n_derivative_relations += 1
        # cardinality is only set for the answer relation.
        if rm.cardinality != -1: # this is a tuple but gets initalized as -1
            # Number of facts in the relation (like in FreebaseEasy).
            cardinality = rm.cardinality[0]

    n_literal_relation_tokens = len(token_name_match_score)
    n_derivation_relation_tokens = len(token_derivation_match_score)
    n_word_relation_tokens = len(token_word_match_score)
    n_weak_relation_tokens = len(token_weak_match_score)
    sum_weak_relation_tokens = round(sum(token_weak_match_score.values()), 2)
    sum_context_relation_tokens = round(sum(token_word_match_score.values()), 6)
    avg_em_surface_score = round(sum(em_surface_scores) / len(em_surface_scores), 2)
    sum_em_surface_score = round(sum(em_surface_scores), 2)
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
    # If we ignore entity features we need to compute coverage differently
    if not entity_features:
        coverage = (n_rel_tokens /
                    float(len(candidate.query.query_tokens)))
    else:
        coverage = ((n_rel_tokens + n_entity_tokens) /
                    float(len(candidate.query.query_tokens)))
    features = {}
    relation_match = 1 if len(candidate.matched_relations) > 0 else 0
    result_size_0 = 1 if result_size == 0 else 0
    matches_answer_type = candidate.matches_answer_type
    if generic_features:
        if entity_features:
            features.update({
                'n_literal_entities': n_literal_entities,
                'n_entity_matches': n_entity_matches,
                'n_text_and_question_entities': n_text_and_question_entities > 0,
                'literal_entities_length': literal_entities_length,
                'avg_em_surface_score': avg_em_surface_score,
                'sum_em_surface_score': sum_em_surface_score,
                'avg_em_popularity': avg_em_popularity,
                'sum_em_popularity': sum_em_popularity,
                'total_literal_length': (literal_entities_length
                                         + n_literal_relations),
            })
        features.update({
            # "Relation Features"
            'n_relations': len(candidate.get_relation_names()),
            'relation_match': relation_match,
            'n_literal_relations': n_literal_relations,
            'n_word_relations': n_word_relations,
            'n_word_weak_relations': n_word_weak_relations,
            'n_derivative_relations': n_derivative_relations,
            'n_literal_relation_tokens': n_literal_relation_tokens,
            'n_derivation_relation_tokens': n_derivation_relation_tokens,
            'n_word_relation_tokens': n_word_relation_tokens,
            'n_weak_relation_tokens': n_weak_relation_tokens,
            'sum_weak_relation_tokens': sum_weak_relation_tokens,
            'sum_context_relation_tokens': sum_context_relation_tokens,
            'cardinality': cardinality,
            # Changed this
            # 'is_mediator': is_mediator,
            # 'em_token_score': em_token_score,
            # "General Features
            'coverage': coverage,
            'matches_answer_type': matches_answer_type,
            'result_size_0': result_size_0,
        })
    return features


def ngram_features(candidate, ngram_dict):
    """Extract ngram features from the single candidate.

    :param candidate:
    :return:
    """
    ngram_features = dict()

    def add_feature(f_name):
        if ngram_dict is None or f_name in ngram_dict:
            ngram_features[f_name] = 1

    relations = sorted(candidate.get_relation_names())
    all_rels = '_'.join(relations)
    n_grams = get_n_grams_features(candidate)
    for ng in n_grams:
        # Ignore ngrams that only consist of stopfwords.
        if set(ng).issubset(N_GRAM_STOPWORDS):
            continue
        f_name = 'rel:%s+word:%s' % (all_rels, '_'.join(ng))
        add_feature(f_name)
    return ngram_features


def split_relation(relation):
    """Split a relation into individual tokens

    :param relation:
    :return:
    """
    return relation.split('.')


def extract_ngram_features(candidates, ngram_dict=None):
    features = []
    for c in candidates:
        features.append(ngram_features(c, ngram_dict))
    return features


def extract_features(candidates,
                     rel_score_model=None,
                     deep_rel_score_model=None,
                     ds_deep_rel_score_model=None):
    """Extract features for multiple candidates at once.

    :param candidates:
    :return:
    """
    all_features = []
    for c in candidates:
        all_features.append(simple_features(c))
    if ds_deep_rel_score_model:
        ds_deep_rel_scores = ds_deep_rel_score_model.score_multiple(candidates)
        for i, f in enumerate(all_features):
            f['ds_deep_relation_scores'] = ds_deep_rel_scores[i].score
    if deep_rel_score_model:
        deep_rel_scores = deep_rel_score_model.score_multiple(candidates)
        for i, f in enumerate(all_features):
            f['deep_relation_score'] = deep_rel_scores[i].score
    if rel_score_model:
        rel_scores = rel_score_model.score_multiple(candidates)
        for i, f in enumerate(all_features):
            f['relation_score'] = rel_scores[i].score
    return all_features
