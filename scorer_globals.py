"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import json
import logging
from query_translator import ranker
from collections import OrderedDict
from entity_linker.entity_linker_qlever import EntityLinkerQlever
from entity_linker.entity_oracle import EntityOracle

free917_entities = "evaluation-data/free917_entities.txt"

logger = logging.getLogger(__name__)

class Conf:
    """
    Holds information about a particular configuration of a scorer and
    allows instantiating the scorer with this configuation, optionally overriding
    some or all parameters.
    """
    def __init__(self, clzz, name, **kwargs):
        self.name = name
        self.clzz = clzz
        self._kwargs = kwargs
        self._override = {}
        self._inst = None

    def config(self):
        """
        Returns the configuration options as a map
        """
        return self._kwargs.copy()

    def override(self):
        """
        Returns only those configuration options that were in the last override
        """
        return self._override.copy()

    def instance(self, override={}):
        """
        Gets an instance of a scorer with this configuration, if one was aleardy created
        a cached one is used
        """
        self._override = override
        newkwargs = self.config()
        newkwargs.update(override)
        if newkwargs != self._kwargs or not self._inst:
            self._kwargs = newkwargs
            logger.info('Instantiating scorer: %s with parameters: %s',
                        self.name, json.dumps(self._kwargs))
            self._inst = self.clzz(self.name, **self._kwargs)
        return self._inst


# The scorers that can be selected.
scorer_list = [Conf(ranker.AqquModel, 'F917_Ranker',
                    train_datasets=["free917train"],
                    top_ngram_percentile=10,
                    rel_regularization_C=1e-6),
               Conf(ranker.AqquModel, 'F917_EQL_Ranker',
                    train_datasets=["free917train"],
                    entity_linker_class=EntityLinkerQlever,
                    top_ngram_percentile=10,
                    rel_regularization_C=1e-6),
               Conf(ranker.AqquModel, 'F917_Ranker_no_types',
                    train_datasets=["free917train"],
                    top_ngram_percentile=10,
                    rel_regularization_C=1e-6,
                    use_type_names=False),
               Conf(ranker.AqquModel, 'F917_Ranker_no_attention',
                    train_datasets=["free917train"],
                    top_ngram_percentile=10,
                    rel_regularization_C=1e-6,
                    use_type_names=True,
                    use_attention=False),
               Conf(ranker.AqquModel, 'F917_Ranker_entity_oracle',
                    train_datasets=["free917train"],
                    entity_oracle_file=free917_entities,
                    entity_linker_class=EntityOracle,
                    top_ngram_percentile=10,
                    rel_regularization_C=1e-6),

               Conf(ranker.AqquModel, 'F917_WQSP_Ranker',
                    train_datasets=["free917train", "wqsptrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'WQSP_F917_Ranker',
                    train_datasets=["wqsptrain", "free917train"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),

               Conf(ranker.AqquModel, 'WQ_Ranker',
                    train_datasets=["webquestionstrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'WQ_Ranker_no_types',
                    train_datasets=["webquestionstrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=False),
               Conf(ranker.AqquModel, 'WQ_Ranker_no_attention',
                    train_datasets=["webquestionstrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=True,
                    use_attention=False),
               Conf(ranker.AqquModel, 'WQ_Ranker_tiny',
                    train_datasets=["webquestionstrain_tiny"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'WQ_Ranker_no_deep',
                    train_datasets=["webquestionstrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    learn_deep_rel_model=False,
                    learn_ngram_rel_model=True),
               Conf(ranker.AqquModel, 'WQ_Ranker_no_deep_1of2',
                    train_datasets=["webquestionstrain_1of2"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    learn_deep_rel_model=False,
                    learn_ngram_rel_model=True),
               Conf(ranker.AqquModel, 'WQ_Ranker_no_deep_gridsearch',
                    train_datasets=["webquestionstrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=None,
                    learn_deep_rel_model=False,
                    learn_ngram_rel_model=True),
               Conf(ranker.AqquModel, 'WQ_Ranker_1of2',
                    train_datasets=["webquestionstrain_1of2"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'WQ_Ranker_EQL',
                    train_datasets=["webquestionstrain"],
                    entity_linker_class=EntityLinkerQlever,
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),

               Conf(ranker.AqquModel, 'SQ_Ranker',
                    train_datasets=["sqtrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'SQ_Ranker_no_types',
                    train_datasets=["sqtrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=False),
               Conf(ranker.AqquModel, 'SQ_Ranker_no_types',
                    train_datasets=["sqtrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=True,
                    use_attention=False),
               Conf(ranker.AqquModel, 'SQ_Ranker_tiny',
                    train_datasets=["sqtrain_tiny"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),

               Conf(ranker.AqquModel, 'WQSP_Ranker',
                    train_datasets=["wqsptrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    num_filters=64,
                    num_hidden_nodes=200),
               Conf(ranker.AqquModel, 'WQSP_Ranker_no_types',
                    train_datasets=["wqsptrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=False),
               Conf(ranker.AqquModel, 'WQSP_Ranker_no_attention',
                    train_datasets=["wqsptrain"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=True,
                    use_attention=False),
               Conf(ranker.AqquModel, 'WQSP_Ranker_no_ngram',
                    train_datasets=["wqsptrain"],
                    rel_regularization_C=1e-5,
                    learn_deep_rel_model=True,
                    learn_ngram_rel_model=False),
               Conf(ranker.AqquModel, 'WQSP_Ranker_EQL',
                    train_datasets=["wqsptrain"],
                    entity_linker_class=EntityLinkerQlever,
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'WQSP_Ranker_tiny',
                    train_datasets=["wqsptrain_tiny"],
                    top_ngram_percentile=10,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'WQSP_Ranker_EQL_tiny',
                    train_datasets=["wqsptrain_tiny"],
                    entity_linker_class=EntityLinkerQlever,
                    top_ngram_percentile=10,
                    rel_regularization_C=1e-5),

               Conf(ranker.AqquModel, 'SQ_WQSP_Ranker_tiny',
                    train_datasets=["sqtrain_tiny", "wqsptrain_tiny"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5),
               Conf(ranker.AqquModel, 'SQ_WQSP_Ranker_tiny_no_types',
                    train_datasets=["sqtrain_tiny", "wqsptrain_tiny"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=False),
               Conf(ranker.AqquModel, 'SQ_WQSP_Ranker_tiny_no_attention',
                    train_datasets=["sqtrain_tiny", "wqsptrain_tiny"],
                    top_ngram_percentile=5,
                    rel_regularization_C=1e-5,
                    use_type_names=True,
                    use_attention=False),

               Conf(ranker.SimpleScoreRanker, 'SimpleRanker'),
               Conf(ranker.SimpleScoreRanker, 'SimpleRanker_entity_oracle',
                                   entity_oracle_file=free917_entities),
               Conf(ranker.LiteralRanker, 'LiteralRanker'),
               Conf(ranker.LiteralRanker,'LiteralRanker_entity_oracle',
                                   entity_oracle_file=free917_entities),
               ]

# A dictionary used for lookup via scorer name.
scorers_dict = OrderedDict(
    [(s.name, s) for s in scorer_list]
)

# A dict of dataset name and file.
DATASETS = OrderedDict(
    [('sqtrain_tiny',
      'evaluation-data/'
      'simple_questions_train_tiny.tsv'),
     ('sqtrain',
      'evaluation-data/'
      'simple_questions_train.tsv'),
     ('sqvalidate',
      'evaluation-data/'
      'simple_questions_valid.tsv'),
     ('free917train',
      'evaluation-data/'
      'free917.train.json'),
     ('webquestionstrain',
      'evaluation-data/'
      'webquestions.train.json'),
     ('wqsptrain',
      'evaluation-data/'
      'WebQSP.train.json'),
     ('wqsptrain_tiny',
      'evaluation-data/'
      'WebQSP_tiny.train.json'),
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
