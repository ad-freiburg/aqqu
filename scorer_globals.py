"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from query_translator import ranker
from collections import OrderedDict
from entity_linker.entity_linker_qlever import EntityLinkerQlever
from entity_linker.entity_oracle import EntityOracle

free917_entities = "evaluation-data/free917_entities.txt"

# The scorers that can be selected.
scorer_list = [ranker.AqquModel('F917_Ranker',
                                "free917train",
                                top_ngram_percentile=10,
                                rel_regularization_C=1e-6),
               ranker.AqquModel('F917_EQL_Ranker',
                                "free917train",
                                entity_linker_class=EntityLinkerQlever,
                                top_ngram_percentile=10,
                                rel_regularization_C=1e-6),
               ranker.AqquModel('F917_Ranker_entity_oracle',
                                "free917train",
                                entity_oracle_file=free917_entities,
                                entity_linker_class=EntityOracle,
                                top_ngram_percentile=10,
                                rel_regularization_C=1e-6),
               ranker.AqquModel('WQ_Ranker',
                                "webquestionstrain",
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5),
               ranker.AqquModel('WQ_Ranker_tiny',
                                "webquestionstrain_tiny",
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5),
               ranker.AqquModel('WQSP_Ranker',
                                "wqsptrain",
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5),
               ranker.AqquModel('WQSP_Ranker_no_ngram',
                                "wqsptrain",
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5,
                                learn_deep_rel_model=True,
                                learn_ngram_rel_model=False),
               ranker.AqquModel('WQSP_EQL_Ranker',
                                "wqsptrain",
                                entity_linker_class=EntityLinkerQlever,
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5),
               ranker.AqquModel('WQ_Ranker_no_deep',
                                "webquestionstrain",
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5,
                                learn_deep_rel_model=False,
                                learn_ngram_rel_model=True),
               ranker.AqquModel('WQ_Ranker_no_deep_1of2',
                                "webquestionstrain_1of2",
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5,
                                learn_deep_rel_model=False,
                                learn_ngram_rel_model=True),
               ranker.AqquModel('WQ_Ranker_no_deep_gridsearch',
                                "webquestionstrain",
                                top_ngram_percentile=5,
                                rel_regularization_C=None,
                                learn_deep_rel_model=False,
                                learn_ngram_rel_model=True),
               ranker.AqquModel('WQ_Ranker_1of2',
                                "webquestionstrain_1of2",
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5),
               ranker.AqquModel('WQ_EQL_Ranker',
                                "webquestionstrain",
                                entity_linker_class=EntityLinkerQlever,
                                top_ngram_percentile=5,
                                rel_regularization_C=1e-5),
               ranker.SimpleScoreRanker('SimpleRanker'),
               ranker.SimpleScoreRanker('SimpleRanker_entity_oracle',
                                        entity_oracle_file=free917_entities),
               ranker.LiteralRanker('LiteralRanker'),
               ranker.LiteralRanker('LiteralRanker_entity_oracle',
                                    entity_oracle_file=free917_entities),
               ]

# A dictionary used for lookup via scorer name.
scorers_dict = OrderedDict(
    [(s.name, s) for s in scorer_list]
)

# A dict of dataset name and file.
DATASETS = OrderedDict(
    [('free917train',
      'evaluation-data/'
      'free917.train.json'),
     ('webquestionstrain',
      'evaluation-data/'
      'webquestions.train.json'),
     ('wqsptrain',
      'evaluation-data/'
      'WebQSP.train.json'),
     ('free917train_1of2',
      'evaluation-data/'
      'free917.train_1of2.json'),
     ('free917train_2of2',
      'evaluation-data/'
      'free917.train_2of2.json'),
     ('webquestionstrain_1of2',
      'evaluation-data/'
      'webquestions.train_1of2.json'),
     ('webquestionstrain_1of2_1of2',
      'evaluation-data/'
      'webquestions.train_1of2_1of2.json'),
     ('webquestionstrain_1of2_2of2',
      'evaluation-data/'
      'webquestions.train_1of2_2of2.json'),
     ('webquestionstrain_2of2',
      'evaluation-data/'
      'webquestions.train_2of2.json'),
     ('webquestionstrain_tiny',
      'evaluation-data/'
      'webquestions.train_tiny.json'),
     ('free917test',
      'evaluation-data/'
      'free917.test.json'),
     ('webquestionstest',
      'evaluation-data/'
      'webquestions.test.json'),
     ('free917test_graphparser',
      'evaluation-data/'
      'free917.test_graphparser.json'),
     ('webquestionstest_graphparser',
      'evaluation-data/'
      'webquestions.test_graphparser.json'),
     ('webquestionstrain_graphparser',
      'evaluation-data/'
      'webquestions.train_graphparser.json'),
     ]
)
