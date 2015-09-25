__author__ = 'haussmae'

import logging

def test_kl_divergence():
    import pattern_matcher
    a = {'a': 10, 'b': 20}
    b = {'a': 20, 'b': 40}
    kl = pattern_matcher.kl_divergence(a, b, smooth=False)
    assert(kl == 0.0)
    b = {'c': 990}
    kl = pattern_matcher.kl_divergence(a, b, smooth=True)
    import data
    rel = 'user.jefft0.default_domain.virus_classification_rank.virus_classifications_at_this_rank'
    word_entity_types = data.read_word_type_distributions('data/word-entity-type-counts_filtered')
    wtypes = word_entity_types['genre']
    target_types = data.read_relation_target_type_distributions('data/relation-target-type-distributions')
    rtypes = target_types[rel]
    wtypes = pattern_matcher.filter_type_distribution(wtypes, n_max=10, min_count=1)
    rtypes = pattern_matcher.filter_type_distribution(rtypes, n_max=10, min_count=1)
    print pattern_matcher.kl_divergence(wtypes, rtypes, alpha=0.1)
    print rtypes
    print wtypes

def test():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(module)s : %(message)s', level=logging.DEBUG)
    test_kl_divergence()

if __name__ == '__main__':
    test()
