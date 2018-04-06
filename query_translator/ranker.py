"""
Classes for scoring and ranking query candidates.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import math
import time
import os
import copy
import logging
import itertools
import random
import numpy as np
import config_helper
from sklearn import utils
from sklearn import metrics
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from entity_linker.entity_linker import EntityLinker
from entity_linker.entity_linker_qlever import EntityLinkerQlever
from entity_linker.entity_oracle import EntityOracle
from sklearn.model_selection import KFold, GridSearchCV
from query_translator.deep_relscorer import DeepCNNAqquRelScorer
from query_translator import feature_extraction as f_ext



RANDOM_SHUFFLE = 0.3

logger = logging.getLogger(__name__)

def Compare2Key(key_func, cmp_func):
    # TODO(schnelle) We should find a refactoring where candidates
    # know how to compare to each other
    """Convert a cmp= function and a key= into a key= function"""
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return cmp_func(key_func(self.obj), key_func(other.obj)) < 0
        def __gt__(self, other):
            return cmp_func(key_func(self.obj), key_func(other.obj)) > 0
        def __eq__(self, other):
            return cmp_func(key_func(self.obj), key_func(other.obj)) == 0
        def __le__(self, other):
            return cmp_func(key_func(self.obj), key_func(other.obj)) <= 0
        def __ge__(self, other):
            return cmp_func(key_func(self.obj), key_func(other.obj)) >= 0
        def __ne__(self, other):
            return cmp_func(key_func(self.obj), key_func(other.obj)) != 0
    return K

class RankScore:
    """A simple score for each candidate.
    """

    def __init__(self, score):
        self.score = score

    def as_string(self):
        return "%s" % self.score

class RankerParameters:
    """A class that holds parameters for the ranker."""

    def __init__(self):
        self.relation_oracle = None
        # When generating candidates, restrict them to the
        # deterimined answer type.
        self.restrict_answer_type = True
        # When matching candidates, require that relations
        # match in some way in the question.
        self.require_relation_match = True
        # Class of the EntityLinker to use
        # one of [EntityLinker, EntityLinkerQlever, EntityOracle]
        self.entity_linker_class = None
        # Path to file containing EntityOracle data,
        # ignored by all other EntityLinkers
        self.entity_oracle_file = None


    def get_suffix(self):
        """Return a suffix string for the selected parameters.

        :type parameters RankerParameters
        :param parameters:
        :return:
        """
        suffix = ""
        if self.entity_linker_class == EntityOracle:
            suffix += "_eo"
        elif self.entity_linker_class == EntityLinkerQlever:
            suffix += "_eql"

        if not self.require_relation_match:
            suffix += "_arm"
        if not self.restrict_answer_type:
            suffix += "_atm"
        return suffix


class Ranker:
    """Superclass for rankers.

    The default is to compute a score for each candidate
    and rank by that score."""

    def __init__(self,
                 name,
                 entity_linker_class=EntityLinker,
                 entity_oracle_file=None,
                 all_relations_match=True,
                 all_types_match=True,
                 **kwargs):  # ignored but used by child classes
        self.name = name
        self.parameters = RankerParameters()
        self.parameters.entity_linker_class = entity_linker_class
        self.parameters.entity_oracle_file = entity_oracle_file
        self.parameters.require_relation_match = not all_relations_match
        self.parameters.restrict_answer_type = not all_types_match

    def get_parameters(self):
        """Return the parameters of the ranker.

        :rtype TranslatorParameters
        :return:
        """
        return self.parameters

    def score(self, candidate):
        """Score each candidate.

        :param candidate:
        :return:
        """
        raise NotImplementedError

    def compare(self, x_candidate, y_candidate):
        """Just compare the ranking scores.

        :param x_candidate:
        :param y_candidate:
        :return:
        """
        x_score = x_candidate.rank_score.score
        y_score = y_candidate.rank_score.score
        return x_score - y_score

    def rank_query_candidates(self, query_candidates, key=lambda x: x,
                              store_features=False):
        """Rank query candidates by scoring and then sorting them.

        :param query_candidates:
        :return:
        """
        query_candidates = shuffle_candidates(query_candidates, key)
        for qc in query_candidates:
            candidate = key(qc)
            candidate.rank_score = self.score(candidate)
        ranked_candidates = sorted(query_candidates,
                                   key=Compare2Key(key, self.compare),
                                   reverse=True)
        return ranked_candidates


class MLModel:
    """Superclass for machine learning based scorer."""

    def __init__(self, name, train_datasets):
        self.name = name
        self.train_datasets = train_datasets

    def get_model_filename(self):
        """Return the model file name."""
        model_base_dir = self.get_model_dir()
        model_file = "%s/%s.model" % (model_base_dir, self.get_model_name())
        return model_file

    def get_model_dir(self):
        """Return the model path."""
        model_base_dir = config_helper.config.get('Ranker', 'model-dir')
        return model_base_dir

    def load_model(self):
        """Loads an MLModel from the model file"""
        pass

    def get_model_name(self):
        """Return the model name."""
        if hasattr(self, "get_parameters"):
            param_suffix = self.get_parameters().get_suffix()
        else:
            param_suffix = ""
        if self.train_datasets:
            model_filename = "%s_%s%s" % (self.name,
                                          '_'.join(self.train_datasets),
                                          param_suffix)
        else:
            model_filename = "%s%s" % (self.name,
                                       param_suffix)
        return model_filename

    def print_model(self):
        """Print info about the model.

        :return:
        """
        pass


class AqquModel(MLModel, Ranker):
    """Performs a pair-wise transform to learn a ranking.

     It always compares two candidates and makes a classification decision
     using a random forest to decide which one should be ranked higher.
    """
    default_config = {
        'top_ngram_percentile': 5,
        'rel_regularization_C': None,
        'learn_deep_rel_model': True,
        'learn_ngram_rel_model': True,
        }

    default_config.update(DeepCNNAqquRelScorer.default_config)

    def score(self, candidate):
        pass

    def __init__(self, name,
                 train_datasets,
                 top_ngram_percentile,
                 rel_regularization_C,
                 learn_deep_rel_model,
                 learn_ngram_rel_model,
                 **kwargs):
        MLModel.__init__(self, name, train_datasets)
        Ranker.__init__(self, name, **kwargs)
        # Note: The model is lazily loaded when score is called.
        self.model = None
        self.label_encoder = None
        self.dict_vec = None
        # The index of the correct label.
        self.correct_index = -1
        self.cmp_cache = dict()
        self.relation_scorer = None
        self.deep_relation_scorer = None
        self.pruner = None
        self.scaler = None
        self.kwargs = kwargs
        self.top_ngram_percentile = top_ngram_percentile
        self.rel_regularization_C = rel_regularization_C
        self.learn_deep_rel_model = learn_deep_rel_model
        self.learn_ngram_rel_model = learn_ngram_rel_model

    def load_model(self):
        model_file = self.get_model_filename()
        try:

            [model, label_enc, dict_vec, pair_dict_vec, scaler] \
                = joblib.load(model_file)
            self.model = model
            self.scaler = scaler
            if self.learn_ngram_rel_model:
                relation_scorer = RelationNgramScorer(
                    self.get_model_name(),
                    self.rel_regularization_C)
                relation_scorer.load_model()
                self.relation_scorer = relation_scorer
            if self.learn_deep_rel_model:
                self.deep_relation_scorer = \
                    DeepCNNAqquRelScorer.init_from_config(
                        **self.kwargs)

                model_dir_tf = os.path.join(self.get_model_dir(), 'tf')
                self.deep_relation_scorer.load_model(
                    model_dir_tf, self.get_model_name())

            self.dict_vec = dict_vec
            self.pair_dict_vec = pair_dict_vec
            pruner = CandidatePruner(self.get_model_name(),
                                     dict_vec)
            pruner.load_model()
            self.pruner = pruner
            self.label_encoder = label_enc
            logger.info("Loaded scorer model from %s" % model_file)
        except IOError:
            logger.warn("Model file %s could not be loaded." % model_file)
            raise

    def learn_rel_score_model(self, queries, ngrams_dict=None):
        rel_model = RelationNgramScorer(self.get_model_name(),
                                        self.rel_regularization_C,
                                        ngrams_dict=ngrams_dict)
        rel_model.learn_model(queries)
        return rel_model

    def learn_deep_rel_score_model(self, queries, test_queries):
        rel_model = DeepCNNAqquRelScorer.init_from_config(
            **self.kwargs)
        extend_deep_model = config_helper.config.get('Ranker',
                                                     'extend-deep-model',
                                                     fallback=None)
        rel_model.learn_model(queries, test_queries,
                              extend_model=extend_deep_model)
        return rel_model

    def learn_prune_model(self, labels, features, dict_vec):
        prune_model = CandidatePruner(self.get_model_name(),
                                      dict_vec)
        prune_model.learn_model(labels, features)
        return prune_model

    def learn_model(self, train_queries):
        f_extract = f_ext.extract_features
        dict_vec = DictVectorizer(sparse=False)
        # Extract features for each candidate onc
        labels, features = construct_train_examples(train_queries,
                                                    f_extract,
                                                    score_threshold=.8)
        features = dict_vec.fit_transform(features)
        n_grams_dict = None
        if self.top_ngram_percentile:
            logger.info("Collecting frequent n-gram features...")
            n_grams_dict = get_top_chi2_candidate_ngrams(train_queries,
                                                         f_ext.extract_ngram_features,
                                                         percentile=self.top_ngram_percentile)
            logger.info("Collected %s n-gram features" % len(n_grams_dict))

        # Compute deep/ngram relation-score based on folds and add
        dict_vec, sub_features = self.learn_submodel_features(train_queries, dict_vec,
                                                              ngrams_dict=n_grams_dict)
        features = np.hstack([features, sub_features])
        if self.learn_ngram_rel_model:
            logger.info("Training relation scorer.")
            rel_model = self.learn_rel_score_model(train_queries, ngrams_dict=n_grams_dict)
            self.relation_scorer = rel_model
        if self.learn_deep_rel_model:
            logger.info("Training deep relation scorer.")
            deep_rel_model = self.learn_deep_rel_score_model(train_queries, None)
            self.deep_relation_scorer = deep_rel_model
        self.dict_vec = dict_vec
        # Pass sparse matrix + dict_vec
        self.pruner = self.learn_prune_model(labels, features, dict_vec)
        self.learn_ranking_model(train_queries, features, dict_vec)

    def learn_ranking_model(self, queries, features, dict_vec):
        # Construct pair examples from whole, pass sparse matrix + train_queries
        pair_dict_vec, pair_features, pair_labels = construct_train_pair_examples(
            queries,
            features,
            dict_vec)
        logger.info("Training tree classifier for ranking.")
        logger.info("#of labeled examples: %s" % len(pair_features))
        logger.info("#labels non-zero: %s" % sum(pair_labels))
        label_encoder = LabelEncoder()
        pair_labels = label_encoder.fit_transform(pair_labels)
        X, labels = utils.shuffle(pair_features, pair_labels, random_state=999)
        decision_tree = RandomForestClassifier(
            random_state=999,
            n_jobs=4,
            n_estimators=200)
        decision_tree.fit(X, labels)
        importances = decision_tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X.shape[1]):
            print("%d. feature %s (%f)" % (f + 1,
                                           pair_dict_vec.feature_names_[indices[f]],
                                           importances[indices[f]]))
        logger.info("Done.")
        self.model = decision_tree
        self.pair_dict_vec = pair_dict_vec
        self.label_encoder = label_encoder


    def store_model(self):
        logger.info("Writing model to %s." % self.get_model_filename())
        joblib.dump([self.model, self.label_encoder,
                     self.dict_vec, self.pair_dict_vec, self.scaler],
                    self.get_model_filename())
        if self.learn_ngram_rel_model:
            self.relation_scorer.store_model()
        if self.learn_deep_rel_model:
            model_dir_tf = os.path.join(self.get_model_dir(), 'tf')
            self.deep_relation_scorer.store_model(
                model_dir_tf, self.get_model_name())
        self.pruner.store_model()
        logger.info("Done.")

    def rank_candidates(self, candidates, features):
        """Pre-compute comparisons.

        The main overhead is calling the classification routine. Therefore,
        pre-computing all O(n^2) comparisons (which can be done with a single
        classification call) is actually faster up to a limit.

        :param candidates:
        :param max_cache_candidates:
        :return:
        """
        if not self.model:
            self.load_model()
        start = time.time()
        num_candidates = len(candidates)
        pairs = list(itertools.combinations(range(num_candidates), 2))
        index_a = [x[0] for x in pairs]
        index_b = [x[1] for x in pairs]
        pair_index = {}
        for i, p in enumerate(pairs):
            pair_index[p] = i
        pair_features = construct_pair_features(features,
                                                np.array(index_a),
                                                np.array(index_b))
        duration = (time.time() - start) * 1000
        logger.info("Constructed %d pair features in %s ms" % (len(pair_features),
                                                                duration))

        X = pair_features
        self.model.n_jobs = 1
        start = time.time()
        #dtest = xgb.DMatrix(X)
        p = self.model.predict(X)
        #p = np.round(p)
        duration = (time.time() - start) * 1000
        logger.info("Predict for %s took %s ms" % (len(pairs), duration))
        #c = p
        c = self.label_encoder.inverse_transform(p)
        start = time.time()
        def compare_pair(i, j):
            if (i, j) in pair_index:
               predict = c[pair_index[(i, j)]]
            else:
                # We only compare i against j, to compare the other direction,
                # j against i, use 1 - p(i, j)
                predict = math.fabs(1 - c[pair_index[(j, i)]])
            if predict == 1:
                return -1
            else:
                return 1
        sorted_i = sorted(range(num_candidates),
                key=Compare2Key(lambda x:x, compare_pair))
        duration = (time.time() - start) * 1000
        logger.info("Sort for %s took %s ms" % (len(pairs), duration))
        return [candidates[i] for i in sorted_i]


    def rank_query_candidates(self, query_candidates, key=lambda x: x,
                              store_features=False):
        """Rank query candidates using the learned comparision function

        :param query_candidates:
        :return:
        """
        if not self.model:
            self.load_model()
        if not query_candidates:
            return []
        query_candidates = shuffle_candidates(query_candidates, key)
        num_candidates = len(query_candidates)
        logger.debug("Pruning %s candidates" % num_candidates)
        start = time.time()
        # Extract features from all candidates and create matrix
        candidates = [key(q) for q in query_candidates]
        features = f_ext.extract_features(candidates,
                                          rel_score_model=self.relation_scorer,
                                          deep_rel_score_model=self.deep_relation_scorer)
        if store_features:
            for i, candidate in enumerate(candidates):
                candidate.feature_dict = features[i]

        features = self.dict_vec.transform(features)
        duration = (time.time() - start) * 1000
        logger.info("Extracted features in %s ms" % (duration))
        query_candidates, features = self.prune_query_candidates(query_candidates,
                                                           features, key)
        logger.info("%s of %s candidates remain" % (len(query_candidates),
                                                    num_candidates))
        start = time.time()
        # If no or only one candidate remains return that..
        if len(query_candidates) < 2:
            return query_candidates
        ranked_candidates = self.rank_candidates(query_candidates,
                                                 features)
        duration = (time.time() - start) * 1000
        logger.debug("Ranked candidates in %s ms" % (duration))
        return ranked_candidates

    def prune_query_candidates(self, query_candidates, features, key=lambda x: x):
        remaining = []
        if len(query_candidates) > 0:
            remaining, new_features = self.pruner.prune_query_candidates(query_candidates,
                                                                         features, key)
        return remaining, new_features

    def learn_submodel_features(self, train_queries, dict_vec, n_folds=6,
                                ngrams_dict=None):
        """Learn additional models based on folds that appear as additional
        features in the final ranking model.

        Return a matrix of additional features + the updated provided dict_vec

        :param train_queries:
        :param dict_vec:
        :return:
        """
        # TODO: could also make learning the "sub-features" the job of the submodels
        # -> have a submodel class. It would be responsible for feature extraction
        # and folding, which would again improve training time if features are
        # extracted only once.
        kf = KFold(n_splits=n_folds, shuffle=True,
                   random_state=999)
        num_fold = 1
        num_features = 2
        # A map form query index to candidate indices
        qc_indices = {}
        qc_index = 0
        for i, q in enumerate(train_queries):
            num_c = len(q.eval_candidates)
            c_indices = [qc_index + c for c in range(num_c)]
            qc_index += num_c
            qc_indices[i] = c_indices
        features = np.zeros(shape=(qc_index, num_features))
        for train_idx, test_idx in kf.split(train_queries):
            logger.info("Training relation score model on fold %s/%s" % (
                num_fold, n_folds))
            test_fold = [train_queries[i] for i in test_idx]
            train_fold = [train_queries[i] for i in train_idx]
            test_candidates = [x.query_candidate for query in test_fold
                               for x in query.eval_candidates]

            rel_scores = []
            if self.learn_ngram_rel_model:
                rel_model = self.learn_rel_score_model(train_fold,
                                                       ngrams_dict=ngrams_dict)
                rel_scores = rel_model.score_multiple(test_candidates)
            deep_rel_scores = []
            if self.learn_deep_rel_model:
                deep_rel_model = self.learn_deep_rel_score_model(train_fold,
                                                                 test_fold)
                deep_rel_scores = deep_rel_model.score_multiple(test_candidates)
            c_index = 0
            for i in test_idx:
                for c in qc_indices[i]:
                    if self.learn_ngram_rel_model:
                        features[c, 0] = rel_scores[c_index].score
                    if self.learn_deep_rel_model:
                        features[c, 1] = deep_rel_scores[c_index].score
                    c_index += 1
            num_fold += 1
        # TODO: better to create a copy and return changed copy
        append_feature_to_dictvec(dict_vec, 'relation_score')
        append_feature_to_dictvec(dict_vec, 'deep_relation_score')
        return dict_vec, features


def append_feature_to_dictvec(dict_vec, feature_name):
    """Append a new feature to the dict vectorizer.

    :param dict_vec:
    :param feature_name:
    :return:
    """
    max_index = max(dict_vec.vocabulary_.values())
    dict_vec.vocabulary_[feature_name] = max_index + 1
    dict_vec.feature_names_.append(feature_name)


class CandidatePruner(MLModel):
    """Learns a recall-optimized pruning model."""

    def __init__(self,
                 name,
                 dict_vec):
        name += self.get_pruner_suffix()
        MLModel.__init__(self, name, None)
        # Note: The model is lazily created when needed.
        self.model = None
        self.label_encoder = None
        self.dict_vec = dict_vec
        self.scaler = None
        # The index of the correct label.
        self.correct_index = -1

    def get_pruner_suffix(self):
        return "_Pruner"

    def print_model(self, n_top=30):
        dict_vec = self.dict_vec
        classifier = self.model
        logger.info("Printing top %s weights." % n_top)
        logger.info("intercept: %.4f" % classifier.intercept_[0])
        feature_weights = []
        for name, index in dict_vec.vocabulary_.items():
            feature_weights.append((name, classifier.coef_[0][index]))
        feature_weights = sorted(feature_weights, key=lambda x: math.fabs(x[1]),
                                 reverse=True)
        for name, weight in feature_weights[:n_top]:
            logger.info("%s: %.4f" % (name, weight))

    def learn_model(self, labels, X):
        logger.info("Learning prune classifier.")
        logger.info("#of labeled examples: %s" % len(X))
        logger.info("#labels non-zero: %s" % sum(labels))
        num_labels = float(len(labels))
        num_pos_labels = sum(labels)
        num_neg_labels = num_labels - num_pos_labels
        pos_class_weight = num_labels / num_pos_labels
        neg_class_weight = num_labels / num_neg_labels
        total_weight = pos_class_weight + neg_class_weight
        pos_class_weight /= total_weight
        neg_class_weight /= total_weight
        # with old ranking 1.0 works best, followed by 1.2
        # with new ranking 1.5 works a lot better
        pos_class_boost = 1.5
        label_encoder = LabelEncoder()
        logger.info(X[-1])
        labels = label_encoder.fit_transform(labels)
        self.label_encoder = label_encoder
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        X, labels = utils.shuffle(X, labels, random_state=999)
        class_weights = {1: pos_class_weight * pos_class_boost,
                         0: neg_class_weight}
        logger.info(class_weights)
        # We want to maximize precision on negative labels
        p_scorer = metrics.make_scorer(metrics.fbeta_score,
                                       pos_label=1, beta=0.5)
        logreg_cv = LogisticRegressionCV(Cs=[1000],
                                         class_weight=class_weights,
                                         cv=3,
                                         solver='sag',
                                         n_jobs=6,
                                         scoring=p_scorer,
                                         # max_iter=40,
                                         verbose=False,
                                         random_state=999)
        logreg_cv.fit(X, labels)
        self.model = logreg_cv
        pred = self.model.predict(X)
        logger.info(logreg_cv.C_)
        logger.info("F-1 score on train: %.4f" % metrics.f1_score(labels, pred,
                                                                  pos_label=1))
        logger.info("Classification report:\n"
                    + classification_report(labels, pred))
        self.label_encoder = label_encoder
        self.print_model()
        logger.info("Done learning prune classifier.")

    def load_model(self):
        model_file = self.get_model_filename()
        try:
            [model, label_enc, scaler] \
                = joblib.load(model_file)
            self.model = model
            self.scaler = scaler
            self.label_encoder = label_enc
            self.correct_index = label_enc.transform([1])[0]
            logger.info("Loaded scorer model from %s" % model_file)
        except IOError:
            logger.warn("Model file %s could not be loaded." % model_file)
            raise

    def store_model(self):
        logger.info("Writing model to %s." % self.get_model_filename())
        joblib.dump([self.model, self.label_encoder,
                     self.scaler], self.get_model_filename())
        logger.info("Done.")

    def prune_query_candidates(self, query_candidates, features, key=lambda x: x):
        remaining = []
        remaining_idxs = []
        X = self.scaler.transform(features)
        p = self.model.predict(X)

        for cand_idx, (cand, predict) in enumerate(zip(query_candidates, p)):
            # TODO the pruner should learn to always prune empty
            # answers but currently it doesn't so check that separately
            if predict == 1 and key(cand).get_result_count() > 0:
                remaining.append(cand)
                remaining_idxs.append(cand_idx)

        new_features = np.zeros(shape=(len(remaining), features.shape[1]))
        for new_idx, cand_idx in enumerate(remaining_idxs):
                new_features[new_idx, :] = features[cand_idx, :]
        return remaining, new_features


class RelationNgramScorer(MLModel):
    """Learns a scoring based on question ngrams."""

    def __init__(self,
                 name,
                 regularization_C,
                 ngrams_dict=None):
        name += self.get_relscorer_suffix()
        MLModel.__init__(self, name, None)
        # Note: The model is lazily when needed.
        self.model = None
        self.regularization_C = regularization_C
        self.ngrams_dict = ngrams_dict
        self.label_encoder = None
        self.dict_vec = None
        self.scaler = None
        # The index of the correct label.
        self.correct_index = -1

    def get_relscorer_suffix(self):
        return "_RelScore"

    def load_model(self):
        model_file = self.get_model_filename()
        try:
            [model, label_enc, dict_vec, scaler] \
                = joblib.load(model_file)
            self.model = model
            self.dict_vec = dict_vec
            self.scaler = scaler
            self.label_encoder = label_enc
            self.correct_index = label_enc.transform([1])[0]
            logger.info("Loaded scorer model from %s" % model_file)
        except IOError:
            logger.warn("Model file %s could not be loaded." % model_file)
            raise

    def test_model(self, test_queries):
        logger.info("Scoring on test fold")
        features, labels = construct_train_examples(test_queries,
                                              f_ext.extract_ngram_features)
        labels = self.label_encoder.transform(labels)
        X = self.dict_vec.transform(features)
        X = self.scaler.transform(X)
        labels_predict = self.model.predict(X)
        logger.info(classification_report(labels, labels_predict))

    def learn_model(self, train_queries):
        def ngram_features(cs):
            return f_ext.extract_ngram_features(cs,
                                                ngram_dict=self.ngrams_dict)

        labels, features = construct_train_examples(train_queries,
                                                    ngram_features)
        logger.info("#of labeled examples: %s" % len(features))
        logger.info("#labels non-zero: %s" % sum(labels))
        num_labels = float(len(labels))
        num_pos_labels = sum(labels)
        num_neg_labels = num_labels - num_pos_labels
        pos_class_weight = num_labels / num_pos_labels
        neg_class_weight = num_labels / num_neg_labels
        total_weight = pos_class_weight + neg_class_weight
        pos_class_weight /= total_weight
        neg_class_weight /= total_weight
        pos_class_boost = 1.0
        label_encoder = LabelEncoder()
        logger.info(features[-1])
        labels = label_encoder.fit_transform(labels)
        vec = DictVectorizer(sparse=True)
        scaler = StandardScaler(with_mean=False)
        X = vec.fit_transform(features)
        X = scaler.fit_transform(X)
        X, labels = utils.shuffle(X, labels, random_state=999)
        logger.info("#Features: %s" % len(vec.vocabulary_))
        class_weights = {1: pos_class_weight * pos_class_boost,
                         0: neg_class_weight} 
        logger.info("Weights: %s" % str(class_weights))
        # Perform grid search or use provided C.
        if self.regularization_C is None:
            logger.info("Performing grid search.")
            # Smaller -> stronger.
            cv_params = [{"C": [1.0, 0.1, 0.01, 0.001, 
                                1e-3, 1e-4, 1e-5, 1e-6]}]
            relation_scorer = LogisticRegression(class_weight=class_weights,
                                                 max_iter=200,
                                                 solver='sag')
            grid_search_cv = GridSearchCV(relation_scorer,
                                          cv_params,
                                          n_jobs=1,
                                          verbose=1,
                                          cv=4,
                                          refit=True,
                                          scoring='roc_auc')
            grid_search_cv.fit(X, labels)
            logger.info("Best score: %.5f" % grid_search_cv.best_score_)
            logger.info("Best params: %s" % grid_search_cv.best_params_)
            self.model = grid_search_cv.best_estimator_
        else:
            logger.info("Learning relation scorer with C: %s."
                        % self.regularization_C)
            # class_weight='balanced' gives 49.15
            # class_weight='auto' gives 49.07
            # class_weight=class_weights gives 49.25 (with 1.0)
            relation_scorer = LogisticRegression(C=self.regularization_C,
                                                 class_weight=class_weights,
                                                 n_jobs=-1,
                                                 solver='sag',
                                                 max_iter=200,
                                                 random_state=999)
            relation_scorer.fit(X, labels)
            logger.info("Done.")
            self.model = relation_scorer
        self.dict_vec = vec
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.correct_index = label_encoder.transform([1])[0]
        #self.print_model()

    def print_model(self, n_top=20):
        dict_vec = self.dict_vec
        classifier = self.model
        logger.info("Printing top %s weights." % n_top)
        logger.info("intercept: %.4f" % classifier.intercept_[0])
        feature_weights = []
        for name, index in dict_vec.vocabulary_.items():
            feature_weights.append((name, classifier.coef_[0][index]))
        feature_weights = sorted(feature_weights, key=lambda x: math.fabs(x[1]),
                                 reverse=True)
        for name, weight in feature_weights[:n_top]:
            logger.info("%s: %.4f" % (name, weight))

    def store_model(self):
        logger.info("Writing model to %s." % self.get_model_filename())
        joblib.dump([self.model, self.label_encoder,
                     self.dict_vec, self.scaler], self.get_model_filename())
        logger.info("Done.")

    def score(self, candidate):
        if not self.model:
            self.load_model()
        features = f_ext.ngram_features(candidate)
        X = self.dict_vec.transform(features)
        X = self.scaler.transform(X)
        prob = self.model.predict_proba(X)
        # Prob is an array of n_examples, n_classes
        score = round(prob[0][self.correct_index], 3)
        return RankScore(score)

    def score_multiple(self, candidates):
        """
        Return a list of scores.
        :param candidates:
        :return:
        """
        if not self.model:
            self.load_model()
        features = f_ext.extract_ngram_features(candidates)
        X = self.dict_vec.transform(features)
        X = self.scaler.transform(X)
        probs = self.model.predict_proba(X)
        # Prob is an array of n_examples, n_classes
        scores = probs[:, self.correct_index]
        return [RankScore(round(score, 3)) for score in scores]


class SimpleScoreRanker(Ranker):
    """Ranks based on a simple score of relation and entity matches."""

    def __init__(self, name, **kwargs):
        Ranker.__init__(self, name, **kwargs)

    def score(self, query_candidate):
        em_token_score = 0.0
        for em in query_candidate.matched_entities:
            em_score = em.entity.surface_score
            em_score *= len(em.entity.tokens)
            em_token_score += em_score
        matched_tokens = dict()
        for rm in query_candidate.matched_relations:
            if rm.name_match:
                for (t, _) in rm.name_match.token_names:
                    matched_tokens[t] = 0.3
            if rm.words_match:
                for (t, s) in rm.words_match.token_scores:
                    if t not in matched_tokens or matched_tokens[t] < s:
                        matched_tokens[t] = s
            if rm.name_weak_match:
                for (t, _, s) in rm.name_weak_match.token_name_scores:
                    s *= 0.1
                    if t not in matched_tokens or matched_tokens[t] < s:
                        matched_tokens[t] = s
        rm_token_score = sum(matched_tokens.values())
        return RankScore(em_token_score + (rm_token_score * 3))


class LiteralRankerFeatures(object):
    """The score object computed by the LiteralScorer.

    Mainly consists of features extracted from each candidate. These
    are used to compare two candidates.
    """

    def __init__(self, ent_lit, rel_lit, coverage,
                 entity_popularity, is_mediator,
                 relation_length, relation_cardinality,
                 entity_score, relation_score,
                 cover_card, result_size):
        self.ent_lit = ent_lit
        self.rel_lit = rel_lit
        self.coverage = coverage
        self.entity_popularity = entity_popularity
        self.is_mediator = is_mediator
        self.relation_length = relation_length
        self.relation_cardinality = relation_cardinality
        self.entity_score = entity_score
        self.relation_score = relation_score
        self.cover_card = cover_card
        self.result_size = result_size

    def as_string(self):
        coverage_weak = self.cover_card - self.coverage
        # lit_max = (self.num_lit == len(self.matched_entities))
        score = self.entity_score + 3 * self.relation_score
        return "ent-lit = %s, rel-lit = %s, cov-lit = %s, cov-weak = %s, " \
               "ent-pop = %.0f, med = %s, rel-len = %s, rel-card = %s, " \
               "score = %.2f, size = %s" % \
               (self.ent_lit,
                self.rel_lit,
                self.coverage,
                coverage_weak,
                self.entity_popularity,
                "yes" if self.is_mediator else "no",
                self.relation_length,
                self.relation_cardinality,
                score,
                self.result_size)

    def rank_query_candidates(self, query_candidates, key=lambda x: x,
                              store_features=False):
        """Rank query candidates by scoring and then sorting them.

        :param query_candidates:
        :return:
        """
        for qc in query_candidates:
            candidate = key(qc)
            candidate.rank_score = self.score(candidate)
            if store_features:
                candidate.feature_dict = {'rank_score': candidate.rank_score}
        ranked_candidates = sorted(query_candidates,
                                   key=Compare2Key(key, self.compare),
                                   reverse=True)
        return ranked_candidates


class LiteralRanker(Ranker):
    """A scorer focusing on literal matches in relations.

    It compares two candidates deciding which of two is better. It uses
    features extracted from both candidates for this. Conceptually, this
    is a simple decision tree.
    """

    def __init__(self, name, **kwargs):
        Ranker.__init__(self, name, **kwargs)

    def score(self, query_candidate):
        """Compute a score object for the candidate.

        :param query_candidate:
        :return:
        """
        literal_entities = 0
        literal_relations = 0
        literal_length = 0
        em_token_score = 0.0
        rm_token_score = 0.0
        em_popularity = 0
        is_mediator = False
        cardinality = 0
        rm_relation_length = 0
        # This is how you can get the size of the result set for a candidate.
        result_size = query_candidate.get_result_count()
        # Each pattern has a name.
        # An "M" indicates a mediator in the pattern.
        if "M" in query_candidate.pattern:
            is_mediator = True
        for em in query_candidate.matched_entities:
            # NEW(Hannah) 22-Mar-15:
            # For entities, also consider strong synonym matches (prob >= 0.8)
            #  as literal matches. This is important for a significant portion
            # of the queries, e.g. "euros" <-> "euro" (prob = 0.998)
            # "protein" <-> "Protein (Nutrients)" (prob = 1.000)
            # "us supreme court" <-> "Supreme Court of the United States"
            # (prob = 0.983) "mozart" <-> "Wolfgang Amadeus Mozart"
            threshold = 0.8
            if em.perfect_match or em.surface_score > threshold:
                literal_entities += 1
                literal_length += len(em.tokens)
            em_score = em.surface_score
            em_score *= len(em.tokens)
            em_token_score += em_score
            if em.score > 0:
                em_popularity += math.log(em.score)
        matched_tokens = dict()
        for rm in query_candidate.matched_relations:
            rm_relation_length += len(rm.relation)
            if rm.name_match:
                literal_relations += 1
                for (t, _) in rm.name_match.token_names:
                    if t not in matched_tokens or matched_tokens[t] < 0.3:
                        literal_length += 1
                        matched_tokens[t] = 0.3
            # Count a match via derivation like a literal match.
            if rm.derivation_match:
                for (t, _) in rm.derivation_match.token_names:
                    if t not in matched_tokens or matched_tokens[t] < 0.3:
                        literal_length += 1
                        matched_tokens[t] = 0.3
            if rm.words_match:
                for (t, s) in rm.words_match.token_scores:
                    if t not in matched_tokens or matched_tokens[t] < s:
                        matched_tokens[t] = s
            if rm.name_weak_match:
                for (t, _, s) in rm.name_weak_match.token_name_scores:
                    s *= 0.1
                    if t not in matched_tokens or matched_tokens[t] < s:
                        matched_tokens[t] = s
            if rm.cardinality != -1: # this was rm.cardinality > 0 but it was a tuple?!?
                # Number of facts in the relation (like in FreebaseEasy).
                cardinality = rm.cardinality[0]
        rm_token_score = sum(matched_tokens.values())
        rm_token_score *= 3.0

        return LiteralRankerFeatures(literal_entities, literal_relations,
                                     literal_length, em_popularity, is_mediator,
                                     rm_relation_length, cardinality,
                                     em_token_score, rm_token_score,
                                     len(query_candidate.covered_tokens()),
                                     result_size)

    def compare(self, x_candidate, y_candidate):
        """Compares two candidates.

        Return 1 iff x should come before y in the ranking, -1 if y should come
        before x, and 0 if the two are equal / their order does not matter.
        """

        # Get the score objects:
        x = x_candidate.rank_score
        y = y_candidate.rank_score

        # For entites, also count strong synonym matches (high "prob") as
        # literal matches, see HannahScorer.score(...) above.
        x_ent_lit = x.ent_lit
        y_ent_lit = y.ent_lit

        # For relations, when comparing a mediator relation to a non-mediator
        # relation, set both to a maximum of 1 (rel_lit = 2 for a mediator
        # relation should not win against rel = 1 for a non-mediator relation).
        x_rel_lit = x.rel_lit
        y_rel_lit = y.rel_lit
        if x.is_mediator != y.is_mediator:
            if x_rel_lit > 1:
                x_rel_lit = 1
            if y_rel_lit > 1:
                y_rel_lit = 1

        # Sum of literal matches and their coverage (see below for an
        # explanation of each). More of this sum is always better.
        tmp = (x_ent_lit + x_rel_lit + x.coverage) - \
                  (y_ent_lit + y_rel_lit + y.coverage)
        if tmp != 0:
            return tmp

        # Literal matches (each entity / relation match counts as one
        #  in num_lit). More of these is always better.
        tmp = (x_ent_lit + x_rel_lit) - (y_ent_lit + y_rel_lit)
        if tmp != 0:
            return tmp

        # Coverage of literal matches (number of questions words covered). More
        # of these is always better, if equal number of literal matches.
        tmp = x.coverage - y.coverage
        if tmp != 0:
            return tmp

        # Coverage of remaining matches (number of questions words
        # covered by weak matches). More of these is always better,
        # if equal number of literal matches and equal coverage of these.
        x_coverage_weak = x.cover_card - x.coverage
        y_coverage_weak = y.cover_card - y.coverage
        assert x_coverage_weak >= 0
        assert y_coverage_weak >= 0
        tmp = x_coverage_weak - y_coverage_weak
        if tmp != 0:
            return tmp

        # Aggregated score of entity and relation match
        # (needed at different points in the two cases that follow).
        x_score = x.entity_score + 3 * x.relation_score
        y_score = y.entity_score + 3 * y.relation_score

        # Now make a case distinction. For all cases, consider the following for
        # tie-breaking (used in various orders in the cases below):
        # - Prefer relations with larger popularity
        # - Prefer non-mediator relations before mediator relations
        # - Prefer relations with shorter string length.
        # - Prefer relations with larger cardinality
        # - For mediator relations: consider cardinality before string length
        # - For non-mediator relations: vice versa.
        # - If everything else is equal: prefer the larger result.
        x_pop_key = x.entity_popularity
        y_pop_key = y.entity_popularity
        x_med_key = 0 if x.is_mediator else 1
        y_med_key = 0 if y.is_mediator else 1
        x_rel_key_1 = -x.relation_length
        x_rel_key_2 = x.relation_cardinality
        y_rel_key_1 = -y.relation_length
        y_rel_key_2 = y.relation_cardinality
        if x.is_mediator:
            x_rel_key_1, x_rel_key_2 = x_rel_key_2, x_rel_key_1
        if y.is_mediator:
            y_rel_key_1, y_rel_key_2 = y_rel_key_2, y_rel_key_1
        x_res_size = x.result_size
        y_res_size = y.result_size

        # CASE 1: same number of literal entity matches and literal relation
        # matches.
        if x_ent_lit == y_ent_lit and x_rel_lit == y_rel_lit:
            # CASE 1.1: at least one literal match (entity or relation).
            # matches (note that values for x and y are the same at this point).
            if x_ent_lit >= 1 or x_rel_lit >= 1:
                # Compare Pareto-style by the listed components. If mediator,
                # consider rel-card before rel-len.
                x_key = (x_pop_key, x_med_key, x_score, x_rel_key_1,
                         x_rel_key_2, x_res_size)
                y_key = (y_pop_key, y_med_key, y_score, y_rel_key_1,
                         y_rel_key_2, y_res_size)
                if x_key < y_key:
                    return -1
                else:
                    return 1

            # CASE 1.2: no literal entity matches and no literal
            # relation matches.
            else:
                # Compare Pareto-style by the listed components. If mediator,
                # consider rel-card before rel-len.
                x_key = (x_score, x.entity_popularity, x_med_key, x_rel_key_1,
                         x_rel_key_2, x_res_size)
                y_key = (y_score, y.entity_popularity, y_med_key, y_rel_key_1,
                         y_rel_key_2, y_res_size)
                if x_key < y_key:
                    return -1
                else:
                    return 1

        # CASE 2: different number of literal entity matches and literal
        # relation matches.
        else:
            # Compare Pareto-style by the listed components.
            x_key = (x_score, x.entity_popularity, x_med_key, x_rel_key_1,
                     x_rel_key_2, x_res_size)
            y_key = (y_score, y.entity_popularity, y_med_key, y_rel_key_1,
                     y_rel_key_2, y_res_size)
            if x_key < y_key:
                return -1
            else:
                return 1


def get_top_chi2_candidate_ngrams(queries, f_extract, percentile):
    """Get top ngrams features according to chi2.
    """
    ngrams_dict = dict()
    labels, features = construct_train_examples(queries, f_extract)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(features)
    # ch2 = SelectKBest(chi2, k=n_features)
    ch2 = SelectPercentile(chi2, percentile=percentile)
    ch2.fit(X, labels)
    indices = ch2.get_support(indices=True)
    for i in indices:
        ngrams_dict[vec.feature_names_[i]] = 1
    return ngrams_dict


def get_compare_indices_for_pairs(queries, correct_threshold):
    compare_indices = []
    candidate_offset = 0
    for query in queries:
        oracle_position = query.oracle_position
        # Only create pairs for which we "know" a correct solution
        # The oracle answer is the one with highest F1 but not necessarily
        # perfect.
        correct_cands_index = set()
        candidates = [x.query_candidate for x in query.eval_candidates]
        for i, _ in enumerate(candidates):
            eval_result = query.eval_candidates[i].evaluation_result
            if i + 1 == oracle_position or \
                    eval_result.f1 >= correct_threshold or \
                    eval_result.parse_match:
                correct_cands_index.add(i)

        if correct_cands_index:
            n_candidates = len(candidates)
            sample_size = n_candidates // 2
            if sample_size < 200:
                sample_size = min(200, n_candidates)
            sample_candidates_index = random.sample(range(n_candidates), sample_size)
            #sample_size = min(100, n_candidates)
            #sample_candidates_index = range(n_candidates)
            for sample_candidate_index in sample_candidates_index:
                for correct_cand_index in correct_cands_index:
                    if sample_candidate_index in correct_cands_index:
                        continue
                    correct_index = correct_cand_index + candidate_offset
                    incorrect_index = sample_candidate_index + candidate_offset
                    compare_indices.append((correct_index, incorrect_index))
        candidate_offset += len(candidates)
    return compare_indices


def construct_train_pair_examples(queries, features, dict_vec,
                                  correct_threshold=.9):
    """Construct training examples from candidates using pair-wise transform.

    :type queries list[EvaluationQuery]
    :return:
    """
    # Create a new matrix of pair examples based on the queries and labels
    # Append one matrix to the other
    # Return the matrix + an updated dict_vec
    logger.info("Extracting ranking features from candidates.")
    # A list of tuples of indices where the element at first index is better.
    compare_indices = get_compare_indices_for_pairs(queries, correct_threshold)
    # Create the feature matrix
    num_compare_examples = len(compare_indices)
    pos_i = [c[0] for c in compare_indices]
    neg_i = [c[1] for c in compare_indices]
    c_pair_features = construct_pair_features(features, pos_i, neg_i)
    i_pair_features = construct_pair_features(features, neg_i, pos_i)
    pair_features = np.vstack([c_pair_features, i_pair_features])
    pair_labels = [1 for _ in range(num_compare_examples)]
    pair_labels += [0 for _ in range(num_compare_examples)]
    # Update the dict_vec
    feature_names = [f + "_a-b" for f in dict_vec.feature_names_]
    feature_names += [f + "_a" for f in dict_vec.feature_names_]
    feature_names += [f + "_b" for f in dict_vec.feature_names_]
    pair_vocab = {f: i for i, f in enumerate(feature_names)}
    # This is a HACK.
    pair_dict_vec = copy.deepcopy(dict_vec)
    pair_dict_vec.feature_names_ = feature_names
    pair_dict_vec.vocabulary_ = pair_vocab
    return pair_dict_vec, pair_features, pair_labels


def construct_pair_features(features, indexes_a, indexes_b):
    """Return features for comparing indexes_a to indexes_b

    :param features:
    :param indexes_a:
    :param indexes_b:
    :return:
    """
    f_a = features[indexes_a, :]
    f_b = features[indexes_b, :]
    examples = np.hstack([f_a - f_b, f_a, f_b])
    return examples


def construct_train_examples(train_queries, f_extract, score_threshold=1.0):
    """Extract features from each candidate.
    Return labels, a matrix of features.

    :param train_queries:
    :return:
    """
    candidates = [x.query_candidate
                  for q in train_queries
                  for x in q.eval_candidates]
    features = f_extract(candidates)
    logger.info("Extracting features from candidates.")
    labels = []
    for query in train_queries:
        oracle_position = query.oracle_position
        candidates = [x.query_candidate for x in query.eval_candidates]
        for i, candidate in enumerate(candidates):
            eval_result = query.eval_candidates[i].evaluation_result
            if i + 1 == oracle_position or \
                    eval_result.f1 >= score_threshold or \
                    eval_result.parse_match:
                labels.append(1)
            else:
                labels.append(0)
    return labels, features


def construct_ngram_examples(queries, f_extractor, correct_threshold=.9):
    """Construct training examples from candidates.

    Construct a list of examples from the given evaluated queries.
    Returns a list of features and a list of corresponding labels
    :type queries list[EvaluationQuery]
    :return:
    """
    logger.info("Extracting features from candidates.")
    labels = []
    features = []
    for query in queries:
        positive_relations = set()
        seen_positive_relations = set()
        oracle_position = query.oracle_position
        candidates = [x.query_candidate for x in query.eval_candidates]
        negative_relations = set()
        for i, candidate in enumerate(candidates):
            relation = " ".join(candidate.get_relation_names())
            eval_result = query.eval_candidates[i].evaluation_result
            if eval_result.f1 == correct_threshold or \
                    i + 1 == oracle_position or \
                    eval_result.parse_match:
                positive_relations.add(relation)
        for i, candidate in enumerate(candidates):
            relation = " ".join(candidate.get_relation_names())
            candidate_features = f_extractor.extract_features(candidate)
            if relation in positive_relations and \
                    relation not in seen_positive_relations:
                seen_positive_relations.add(relation)
                labels.append(1)
                features.append(candidate_features)
            elif relation not in negative_relations:
                negative_relations.add(relation)
                labels.append(0)
                features.append(candidate_features)
    return features, labels


def feature_diff(features_a, features_b):
    """Compute features_a - features_b

    :param features_a:
    :param features_b:
    :return:
    """
    keys = set(itertools.chain(features_a.keys(), features_b.keys()))
    diff = dict()
    for k in keys:
        v_a = features_a.get(k, 0.0)
        v_b = features_b.get(k, 0.0)
        diff[k + "_a"] = v_a
        diff[k + "_b"] = v_b
        diff[k] = v_a - v_b
    return diff


def sort_query_candidates(candidates, key):
    """
    To guarantee consistent results we need to make sure the candidates are
    provided in identical order.
    :param candidates:
    :return:
    """
    candidates = sorted(candidates, key=lambda qc: key(qc).to_sparql_query())
    return candidates


def shuffle_candidates(candidates, key):
    """
    Randomly shuffle the candidates, but make the function idempotent and
    repducible.
    :param candidates:
    :param key:
    :return:
    """
    stable_candidates = sort_query_candidates(candidates, key)
    random.Random(RANDOM_SHUFFLE).shuffle(stable_candidates)
    return stable_candidates
