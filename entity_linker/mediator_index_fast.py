"""A faster inverted index to find entities linked via mediators.

This is a drop in replacement for the existing mediator index. It is about
10x as fast (at least).
Details about the index are in the corresponding Cython file
mediator_index_c.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import logging
import bisect
import mmap
import time
import globals
from mediator_index_c import write_index, read_index, compute_intersection, \
    compute_intersection_for_list, compute_intersection_fast
import numpy as np

logger = logging.getLogger(__name__)
REVERSE_SUFFIX = "_REVERSE"

class MediatorIndexFast(object):

    def __init__(self, index_file_prefix, facts_file):
        self.next_vocab_id = 0
        # The index is just a numpy array of posting lists.
        self.index = None
        # The vocabulary is a list of words.
        self.vocabulary_words = None
        self.reverse_vocab = None
        self.offsets = None
        self.sizes = None
        self.get_or_create_index(index_file_prefix,
                                 facts_file)
        self.cache = {}

    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = globals.config
        mediator_index_prefix = config_options.get('MediatorIndex',
                                                   'mediator-index-prefix')
        mediator_facts = config_options.get('MediatorIndex',
                                            'mediator-facts')
        return MediatorIndexFast(mediator_index_prefix, mediator_facts)

    def get_or_create_index(self, index_file_prefix, facts_file):
        try:
            self.open_index(index_file_prefix)
        except:
            self.build_index(index_file_prefix, facts_file)

    def get_id_for_word(self, word):
        word = word.encode('utf-8')
        insert = bisect.bisect_left(self.vocabulary_words, word)
        if self.vocabulary_words[insert] != word:
            return None
        else:
            return insert

    def open_index(self, index_prefix):
        logger.info("Trying to load index from %s. " % index_prefix)
        try:
            handle, vocabulary_words, \
                offsets, sizes = read_index(index_prefix)
            self.index = handle
            self.offsets = offsets
            self.vocabulary_words = vocabulary_words
            self.sizes = sizes
            logger.info("Index loaded.")
        except IOError:
            logger.info("No existing index found.")
            raise RuntimeError("No existing index.")

    def build_index(self, index_file_prefix, facts_file):
        logger.info("Building new mediator index.")
        num_lines = 0
        vocabulary = {}
        entity_postings = {}
        # Read the vocabulary.
        logger.info("Building vocabulary.")
        vocab__words_set = set()
        with open(facts_file, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            line = mm.readline()
            while line:
                cols = line.strip().split('\t')
                if len(cols) != 4:
                    logger.warn("Invalid line: %s" % line)
                    line = mm.readline()
                    num_lines += 1
                    continue
                cols = [globals.remove_freebase_ns(x) for x in cols]
                vocab__words_set.update(cols)
                line = mm.readline()
                num_lines += 1
                if num_lines % 2000000 == 0:
                    logger.info("Processed %s lines." % num_lines)
        vocabulary_words = sorted(vocab__words_set)
        # This is only for fast reading.
        vocabulary = dict()
        for i, word in enumerate(vocabulary_words):
            vocabulary[word] = i
        # Second pass, this time with vocabulary.
        logger.info("Building index.")
        num_lines = 0
        with open(facts_file, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            line = mm.readline()
            while line:
                cols = line.strip().split('\t')
                if len(cols) != 4:
                    logger.warn("Invalid line: %s" % line)
                    line = mm.readline()
                    num_lines += 1
                    continue
                cols = [globals.remove_freebase_ns(x) for x in cols]
                value_id = vocabulary[cols[0]]
                relation_id = vocabulary[cols[1]]
                mediator_id = vocabulary[cols[3]]
                if value_id not in entity_postings:
                    entity_postings[value_id] = []
                entity_postings[value_id].append((mediator_id, relation_id))
                line = mm.readline()
                num_lines += 1
                if num_lines % 2000000 == 0:
                    logger.info("Processed %s lines." % num_lines)
        logger.info("Sorting postings...")
        for k, v in entity_postings.iteritems():
            a = sorted(entity_postings[k])
            # Remove the tuples
            a = [x for y in a for x in y]
            entity_postings[k] = np.array(a, dtype=np.uint32)
        total_postings = sum([len(x) for _, x in entity_postings.iteritems()])
        logger.info("Number of posting lists: %s " % len(entity_postings))
        logger.info("Avg. posting list length: %s " % (total_postings
                                                       / float(len(entity_postings))))
        logger.info("Writing index.")
        index_handle, offsets, sizes = write_index(index_file_prefix,
                                                   vocabulary_words, entity_postings)
        self.vocabulary_words = vocabulary_words
        self.index = index_handle
        self.offsets = offsets
        self.sizes = sizes

    def get_freebase_mediators(self, fb_mid_a, fb_mid_b):
        '''
        Return a list of tuples:
        ('mediator-mid', 'rel_a', 'rel_b') where where rel_x
        is the relation from the mediator to the respective entity.
        If IDs are unknown an empty list is returned.
        :param fb_mid_a:
        :param fb_mid_b:
        :return:
        '''
        id_a = self.get_id_for_word(fb_mid_a)
        id_b = self.get_id_for_word(fb_mid_b)
        if not id_a or not id_b:
            logger.debug("Freebase MID is unknown: %s or %s." % (fb_mid_a,
                                                                 fb_mid_b))
            return []
        intersection = compute_intersection_fast(id_a, id_b, self.index,
                                                 self.offsets, self.sizes)
        result = []
        for (mediator_id, relid_a, relid_b) in intersection:
            result.append((self.vocabulary_words[mediator_id],
                           self.vocabulary_words[relid_a],
                           self.vocabulary_words[relid_b]))
        return result

    def get_freebase_mediators_list(self, fb_mids):
        '''
        Return a list of tuples:
        ('mediator-mid', 'rel_a', 'rel_b') where where rel_x
        is the relation from the mediator to the respective entity.
        If IDs are unknown an empty list is returned.
        :param fb_mid_a:
        :param fb_mid_b:
        :return:
        '''
        vocab_ids = []
        for mid in fb_mids:
            id = self.get_id_for_word(mid)
            if not id:
                logger.debug("Unknown Freebase MID: %s." % mid)
                return []
            vocab_ids.append(id)
        if len(vocab_ids) > 2:
            intersection = compute_intersection_for_list(vocab_ids,
                                                         self.index,
                                                         self.offsets,
                                                         self.sizes)
            result = []
            for result_posting in intersection:
                result.append(tuple([self.vocabulary_words[r]
                                     for r in result_posting]))
            return result
        else:
            logger.warn("Need more than 2 mids.")
            return []


def build_test():
    index = MediatorIndexFast('local-data/mediator_index_fast',
                          'local-data/mediator_facts_clean.txt')
    print index.get_freebase_mediators("m.01z0kvv", "m.01z0kvv")
    print index.get_freebase_mediators_list(["m.02xg9mh", "m.01z0kvl", "m.01z0kvv"])
    import random
    ids = [i for i in range(len(index.vocabulary_words))]
    n_queries = 1000000
    n_successful = 0
    logger.info("Performing %s queries." % n_queries)
    ids_a = []
    ids_b = []
    for _ in range(n_queries):
        ids_a.append(random.choice(ids))
        ids_b.append(random.choice(ids))
    start = time.time()
    for id_a, id_b in zip(ids_a, ids_b):
        id_a = index.vocabulary_words[id_a]
        id_b = index.vocabulary_words[id_b]
        res = index.get_freebase_mediators(id_a, id_b)
        if len(res) > 0:
            n_successful += 1
    duration = (time.time() - start) * 1000
    logger.info("Done in %s ms." % duration)
    logger.info("%s ms per query." % (duration / float(n_queries)))
    logger.info("%s queries with results." % n_successful)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(module)s : %(message)s', level=logging.DEBUG)
    build_test()