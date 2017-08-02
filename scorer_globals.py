"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from query_translator import ranker
from collections import OrderedDict

free917_entities = "evaluation-data/free917_entities.txt"

# The scorers that can be selected.
scorer_list = [ranker.AccuModel('F917_Ranker',
                                 "free917train",
                                 top_ngram_percentile=2,
                                 rel_regularization_C=0.002),
               ranker.AccuModel('F917_EQL_Ranker',
                                 "free917train",
                                 top_ngram_percentile=2,
                                 rel_regularization_C=0.002),
               ranker.AccuModel('F917_Ranker_entity_oracle',
                                 "free917train",
                                 entity_oracle_file=free917_entities,
                                 top_ngram_percentile=2,
                                 rel_regularization_C=1.0),
               ranker.AccuModel('WQ_Ranker',
                                 "webquestionstrain",
                                 top_ngram_percentile=5,
                                 rel_regularization_C=1.0),
               ranker.AccuModel('WQ_EQL_Ranker',
                                 "webquestionstrain",
                                 top_ngram_percentile=5,
                                 rel_regularization_C=1.0),
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
