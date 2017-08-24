"""
A set of methods for reading data files.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
__author__ = 'haussmae'
import freebase
import re
import logging

logger = logging.getLogger(__name__)


def read_relation_expected_types(target_types_file):
    relation_target_types = {}
    with open(target_types_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            relation_name = cols[0]
            type_name = cols[1]
            relation_target_types[relation_name] = type_name
    return relation_target_types


def read_relation_lemmas(lemma_file):
    """Read the lemmatized version of each relation.

    :param lemma_file:
    :return:
    """
    logger.info("Reading relation lemmas from %s." % lemma_file)
    relation_lemma = dict()
    with open(lemma_file, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            name = cols[0]
            lemma_name = cols[1]
            relation_lemma[name] = lemma_name
    logger.info("Read %s relation lemmas." % len(relation_lemma))
    return relation_lemma


def read_word_type_distributions(word_type_counts_file,
                                 min_count=200,
                                 min_type_count=10):
    """Read the word entity type counts.

    The filtering parameters are really only for keeping the data footprint
    small and (should/are meant to) have no other effect.
    :param word_type_counts_file:
    :param min_count: The minimum number of times we must have seen
    the word to consider it.
    :return:
    """
    word_type_counts = dict()
    logger.info("Reading word type counts from %s." % word_type_counts_file)
    with open(word_type_counts_file, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            word = cols[0]
            word_total_count = int(cols[1])
            if word_total_count < min_count:
                continue
            types = cols[2].split(' ')
            word_type_counts[word] = dict()
            for t in types:
                ind = t.rindex(':')
                type_name = t[:ind]
                count = int(t[ind + 1:])
                if count < min_type_count:
                    continue
                word_type_counts[word][type_name] = count
    logger.info("Read type counts for %s words." % len(word_type_counts))
    return word_type_counts


def read_word_derivations(derivations_file):
    """Read the word derivations from a file.

    Returns a dictionary from word to a set of words which are its
    derivations.
    :param derivations_file:
    :return:
    """
    word_derivations = dict()
    logger.info("Reading word derivations from %s." % derivations_file)
    with open(derivations_file, "r", encoding = 'utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            word = cols[0]
            derivations = set(cols[1].split(' '))
            word_derivations[word] = derivations
    logger.info("Read derivations for %s words." % len(word_derivations))
    return word_derivations


def read_relation_target_type_distributions(target_types_file):
    relation_target_types = {}
    ignore_types = {'common.topic'}
    with open(target_types_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            relation_name = cols[0]
            count = int(cols[1])
            types = cols[2].split(' ')
            relation_target_types[relation_name] = dict()
            for t in types:
                if t in ignore_types:
                    continue
                type_name = t[:t.rindex(':')]
                if type_name in ignore_types:
                    continue
                type_count = float(t[t.rindex(':') + 1:])
                relation_target_types[relation_name][type_name] = type_count
            # Avoid empty dicts if all types were ignored.
            if not relation_target_types[relation_name]:
                del relation_target_types[relation_name]
    return relation_target_types


def read_relation_words(relation_words_file, n_top_words=30):
    """
    Read the top relation indicator words from file.
    Return a map from relation -> {word -> score}
    :param relation_words_file:
    :return:
    """
    word_score_re = re.compile(r'([^(]+)\(([^\)]+)\)')
    relation_words = {}
    logging.info("Reading relation words from %s ." % relation_words_file)
    with open(relation_words_file, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            relation_name = cols[0]
            top_words = []
            words = cols[2].split(' ')
            for c in words[:n_top_words]:
                m = re.match(word_score_re, c)
                if m:
                    word = m.group(1)
                    score = float(m.group(2))
                    top_words.append((word, score))
                else:
                    logger.warn("Unknown word format in "
                                "relation words list: %s." % line)
            # Normalize the scores.
            top_words_dict = normalize_word_scores(top_words)
            relation_words[relation_name] = top_words_dict
    logging.info("Read words for %s relations." % len(relation_words))
    return relation_words


def normalize_word_scores(word_scores):
    """
    Returns a dict from word -> normalized score
    :param word_scores: A list of tuples (word, score)
    :return:
    """
    total_score = float(sum([s for w, s in word_scores]))
    return {w: s / total_score for w, s in word_scores}


def read_mediated_relation_words(relation_words_file, n_top_words=10):
    """
    Read the top relation indicator words from file.
    Return a map from relation -> {word -> score}
    :param relation_words_file:
    :return:
    """
    word_score_re = re.compile(r'([^(]+)\(([^\)]+)\)')
    relation_words = {}
    logging.info("Reading mediated relation words from %s ." %
                 relation_words_file)
    with open(relation_words_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            relation_names = tuple(cols[0].split(' '))
            top_words = []
            words = cols[2].split(' ')
            for c in words[:n_top_words]:
                m = re.match(word_score_re, c)
                if m:
                    word = m.group(1)
                    score = float(m.group(2))
                    top_words.append((word, score))
            top_words_dict = normalize_word_scores(top_words)
            relation_words[relation_names] = top_words_dict
    logging.info("Read words for %s mediated relations." % len(relation_words))
    return relation_words


def read_reverse_relations(reverse_relations_file):
    """
    Read the reverse relations file and return a dict.
    :param reverse_relations_file:
    :return:
    """
    prefix = "<http://rdf.freebase.com/ns"
    reverse_relations = {}
    with open(reverse_relations_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            name = cols[0]
            reverse = cols[1]
            reverse_relations[name] = reverse
            reverse_relations[reverse] = name
    return reverse_relations


def read_relation_counts(name_mapping_file):
    """
    Read the relation counts file and return a dict from
    relation -> (total, unique_subjects, unique_objects)
    :param name_mapping_file:
    :return:
    """
    relation_counts = {}
    logger.info("Reading relation counts from %s." % name_mapping_file)
    with open(name_mapping_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            name = cols[0]
            # Some weird 1-character relations...
            if len(name) > 1:
                # Ignore counts of key relations
                if name.startswith(freebase.FREEBASE_KEY_PREFIX):
                    continue
                # name = name[1:].replace('/', '.')
                counts = (int(cols[1]), int(cols[2]), int(cols[3]))
                relation_counts[name] = counts
    logger.info("Read relation counts for %s relations." %
                len(relation_counts))
    return relation_counts


def read_mediator_names(mediator_names_file):
    """
    Read the mediator names file and return a map from
    mid to name.
    :param name_mapping_file:
    :return:
    """
    mediator_names = {}
    with open(mediator_names_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            mid, name = line.strip().split('\t')
            mediator_names[name] = mid
    return mediator_names


def read_mediator_relations(mediator_relations_file):
    """
    Read the mediator names file and return a set mediator
    relations.
    :param name_mapping_file:
    :return:
    """
    mediator_relations = set()
    with open(mediator_relations_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            relation = line.strip()
            mediator_relations.add(relation)
    return mediator_relations
