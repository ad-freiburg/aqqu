"""
Classes for scoring and ranking query candidates.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import math
import time
import logging
from itertools import chain
from . import translator
import random
import globals
import numpy as np
from random import Random
from sklearn import utils
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    AdaBoostRegressor, RandomForestRegressor, ExtraTreesClassifier
from sklearn import pipeline
from sklearn.linear_model import SGDClassifier, SGDRegressor, \
    LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, \
    Normalizer, MinMaxScaler
from .evaluation import EvaluationQuery, EvaluationCandidate
from query_translator.oracle import EntityOracle
from .features import FeatureExtractor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import KFold, GridSearchCV


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

class RankScore(object):
    """A simple score for each candidate.
    """

    def __init__(self, score):
        self.score = score

    def as_string(self):
        return "%s" % self.score


class Ranker(object):
    """Superclass for rankers.

    The default is to compute a score for each candidate
    and rank by that score."""

    def __init__(self,
                 name,
                 entity_oracle_file=None,
                 all_relations_match=True,
                 all_types_match=True):
        self.name = name
        self.parameters = translator.TranslatorParameters()
        if entity_oracle_file:
            self.parameters.entity_oracle = EntityOracle(entity_oracle_file)
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

    def rank_query_candidates(self, query_candidates, key=lambda x: x):
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


class MLModel(object):
    """Superclass for machine learning based scorer."""

    def __init__(self, name, train_dataset):
        self.name = name
        self.train_dataset = train_dataset

    def get_model_filename(self):
        """Return the model file name."""
        model_filename = self.get_model_name()
        model_base_dir = globals.config.get('Ranker', 'model-dir')
        model_file = "%s/%s.model" % (model_base_dir, model_filename)
        return model_file

    def get_model_name(self):
        """Return the model name."""
        if hasattr(self, "get_parameters"):
            param_suffix = translator.get_suffix_for_params(
                self.get_parameters())
        else:
            param_suffix = ""
        if self.train_dataset is not None:
            model_filename = "%s_%s%s" % (self.name,
                                          self.train_dataset,
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


class AccuModel(MLModel, Ranker):
    """Performs a pair-wise transform to learn a ranking.

     It always compares two candidates and makes a classification decision
     using a random forest to decide which one should be ranked higher.
    """

    def score(self, candidate):
        pass

    def __init__(self, name,
                 train_dataset,
                 top_ngram_percentile=5,
                 rel_regularization_C=None,
                 **kwargs):
        MLModel.__init__(self, name, train_dataset)
        Ranker.__init__(self, name, **kwargs)
        # Note: The model is lazily loaded when score is called.
        self.model = None
        self.label_encoder = None
        self.dict_vec = None
        # The index of the correct label.
        self.correct_index = -1
        self.cmp_cache = dict()
        self.relation_scorer = None
        self.pruner = None
        self.scaler = None
        self.kwargs = kwargs
        self.top_ngram_percentile = top_ngram_percentile
        self.rel_regularization_C = rel_regularization_C
        # Only extract ngram features.
        self.feature_extractor = FeatureExtractor(True,
                                                  False,
                                                  None)

    def load_model(self):
        model_file = self.get_model_filename()
        try:

            [model, label_enc, dict_vec, scaler] \
                = joblib.load(model_file)
            self.model = model
            self.scaler = scaler
            relation_scorer = RelationNgramScorer(self.get_model_name(),
                                                  self.rel_regularization_C,
                                                  percentile=self.top_ngram_percentile)
            relation_scorer.load_model()
            self.feature_extractor.relation_score_model = relation_scorer
            pruner = CandidatePruner(self.get_model_name(),
                                     relation_scorer)
            pruner.load_model()
            self.pruner = pruner
            self.dict_vec = dict_vec
            self.label_encoder = label_enc
            self.correct_index = label_enc.transform([1])[0]
            logger.info("Loaded scorer model from %s" % model_file)
        except IOError:
            logger.warn("Model file %s could not be loaded." % model_file)
            raise

    def learn_rel_score_model(self, queries):
        rel_model = RelationNgramScorer(self.get_model_name(),
                                        self.rel_regularization_C,
                                        percentile=self.top_ngram_percentile)
        rel_model.learn_model(queries)
        return rel_model

    def learn_prune_model(self, labels, features):
        prune_model = CandidatePruner(self.get_model_name(),
                                      self.relation_scorer)
        prune_model.learn_model(labels, features)
        return prune_model

    def learn_model(self, train_queries, n_folds=6):
        # split the training queries into folds
        # for each fold extract n-gram features (and select best ones)
        # also extract regular features
        # learn the relation classifier and score the "test" fold
        # add the score as feature in the test-fold
        # collect all test-folds
        # train the treepair classifier on the collected test-folds
        # train the relation classifier on the all relation-features

        kf = KFold(n_splits=n_folds, shuffle=True,
                   random_state=999)
        num_fold = 1
        pair_features = []
        pair_labels = []
        features = []
        labels = []
        for train_indices, test_indices in kf.split(train_queries):
            logger.info("Training relation score model on fold %s/%s" % (
                num_fold, n_folds))
            test_fold = [train_queries[i] for i in test_indices]
            train_fold = [train_queries[i] for i in train_indices]
            rel_model = self.learn_rel_score_model(train_fold)
            self.feature_extractor.relation_score_model = rel_model
            logger.info("Applying relation score model.")
            testfoldpair_features, testfoldpair_labels = construct_pair_examples(
                test_fold,
                self.feature_extractor)
            testfold_features, testfold_labels = construct_examples(
                test_fold,
                self.feature_extractor)
            features.extend(testfold_features)
            labels.extend(testfold_labels)
            pair_features.extend(testfoldpair_features)
            pair_labels.extend(testfoldpair_labels)
            num_fold += 1
            logger.info("Done collecting features for fold.")
        logger.info("Training final relation scorer.")
        rel_model = self.learn_rel_score_model(train_queries)
        self.feature_extractor.relation_score_model = rel_model
        self.relation_scorer = rel_model
        self.pruner = self.learn_prune_model(labels, features)
        self.learn_ranking_model(pair_features, pair_labels)

    def learn_ranking_model(self, features, labels):
        logger.info("Training tree classifier for ranking.")
        logger.info("#of labeled examples: %s" % len(features))
        logger.info("#labels non-zero: %s" % sum(labels))
        label_encoder = LabelEncoder()
        logger.info(features[-1])
        labels = label_encoder.fit_transform(labels)
        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(features)
        X, labels = utils.shuffle(X, labels, random_state=999)
        decision_tree = RandomForestClassifier(class_weight='balanced',
                                               random_state=999,
                                               n_jobs=6,
                                               n_estimators=90)
        logger.info("Training random forest...")
        decision_tree.fit(X, labels)
        logger.info("Done.")
        self.model = decision_tree
        self.dict_vec = vec
        self.label_encoder = label_encoder
        self.correct_index = label_encoder.transform([1])[0]

    def store_model(self):
        logger.info("Writing model to %s." % self.get_model_filename())
        joblib.dump([self.model, self.label_encoder,
                     self.dict_vec, self.scaler],
                    self.get_model_filename())
        self.relation_scorer.store_model()
        self.pruner.store_model()
        logger.info("Done.")

    def compare_pair(self, x_candidate, y_candidate):
        """Compare two candidates.

        Return 1 if x_candidate > y_candidate, else return -1.
        :param x_candidate:
        :param y_candidate:
        :return:
        """
        if not self.model:
            self.load_model()
        # Use the preferences for sorting.
        else:
            res = None
            if (x_candidate, y_candidate) in self.cmp_cache:
                res = self.cmp_cache[(x_candidate, y_candidate)]
            if res is None:
                x_features = self.feature_extractor.extract_features(
                    x_candidate)
                y_features = self.feature_extractor.extract_features(
                    y_candidate)
                diff = feature_diff(x_features, y_features)
                X = self.dict_vec.transform(diff)
                if self.scaler:
                    X = self.scaler.transform(X)
                self.model.n_jobs = 1
                p = self.model.predict(X)
                c = self.label_encoder.inverse_transform(p)
                res = c[0]
            if res == 1:
                return 1
            else:
                return -1

    def _precompute_cmp(self, candidates, max_cache_candidates=300):
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
        self.cmp_cache = dict()
        pairs = []
        pair_features = []
        features = []
        if len(candidates) > max_cache_candidates:
            logger.info("Cannot precoumpte for  all of %s candidates."
                        % len(candidates))
            return
        start = time.time()
        for c in candidates[:max_cache_candidates]:
            f = self.feature_extractor.extract_features(c)
            features.append(f)
        duration = (time.time() - start) * 1000
        logger.debug("FExtract took %s ms" % duration)
        start = time.time()
        for i, x in enumerate(candidates[:max_cache_candidates]):
            x_f = features[i]
            for j, y in enumerate(candidates[:max_cache_candidates]):
                if i == y:
                    continue
                y_f = features[j]
                diff = feature_diff(x_f, y_f)
                pair_features.append(diff)
                pairs.append((x, y))
        duration = (time.time() - start) * 1000
        logger.debug("FDiff for %s took %s ms" % (len(pairs), duration))
        if len(pairs) > 0:
            X = self.dict_vec.transform(pair_features)
            if self.scaler:
                X = self.scaler.transform(X)
            self.model.n_jobs = 1
            start = time.time()
            p = self.model.predict(X)
            duration = (time.time() - start) * 1000
            logger.debug("Predict for %s took %s ms" % (len(pairs), duration))
            self.model.n_jobs = 1
            c = self.label_encoder.inverse_transform(p)
            # Remember the #of wins/losses for each candidate.
            for (x, y), s in zip(pairs, c):
                self.cmp_cache[(x, y)] = s

    def rank_query_candidates(self, query_candidates, key = lambda x: x):
        """Rank query candidates by scoring and then sorting them.

        :param query_candidates:
        :return:
        """
        if not self.model:
            self.load_model()
        query_candidates = shuffle_candidates(query_candidates, key)
        num_candidates = len(query_candidates)
        logger.debug("Pruning %s candidates" % num_candidates)
        query_candidates = self.prune_candidates(query_candidates, key)
        logger.debug("%s of %s candidates remain" % (len(query_candidates),
                                                    num_candidates))
        start = time.time()
        candidates = [key(q) for q in query_candidates]
        self._precompute_cmp(candidates)
        ranked_candidates = sorted(query_candidates,
                                   key=Compare2Key(key, self.compare_pair),
                                   reverse=True)
        self.cmp_cache = dict()
        if len(query_candidates) > 0:
            duration = (time.time() - start) * 1000
            logger.debug(
                "Sorting %s candidates took %s ms. %s ms per candidate" %
                (len(query_candidates), duration,
                 float(duration) / len(query_candidates)))
        return ranked_candidates

    def prune_candidates(self, query_candidates, key):
        remaining = []
        if len(query_candidates) > 0:
            remaining = self.pruner.prune_candidates(query_candidates, key)
        return remaining


class CandidatePruner(MLModel):
    """Learns a recall-optimized pruning model."""

    def __init__(self,
                 name,
                 rel_score_model):
        name += self.get_pruner_suffix()
        MLModel.__init__(self, name, None)
        # Note: The model is lazily when needed.
        self.model = None
        self.label_encoder = None
        self.dict_vec = None
        self.scaler = None
        # The index of the correct label.
        self.correct_index = -1
        self.feature_extractor = FeatureExtractor(True,
                                                  False,
                                                  relation_score_model=rel_score_model,
                                                  entity_features=True)

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

    def learn_model(self, labels, features):
        logger.info("Learning prune classifier.")
        logger.info("#of labeled examples: %s" % len(features))
        logger.info("#labels non-zero: %s" % sum(labels))
        num_labels = float(len(labels))
        num_pos_labels = sum(labels)
        num_neg_labels = num_labels - num_pos_labels
        pos_class_weight = num_labels / num_pos_labels
        neg_class_weight = num_labels / num_neg_labels
        pos_class_boost = 2.0
        label_encoder = LabelEncoder()
        logger.info(features[-1])
        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(features)
        labels = label_encoder.fit_transform(labels)
        self.label_encoder = label_encoder
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        X, labels = utils.shuffle(X, labels, random_state=999)
        class_weights = {1: pos_class_weight * pos_class_boost,
                         0: neg_class_weight}
        logger.info(class_weights)
        logreg_cv = LogisticRegressionCV(Cs=20,
                                         class_weight=class_weights,
                                         cv=6,
                                         solver='lbfgs',
                                         n_jobs=6,
                                         # max_iter=40,
                                         verbose=True)
        logreg_cv.fit(X, labels)
        self.model = logreg_cv
        pred = self.model.predict(X)
        logger.info("F-1 score on train: %.4f" % metrics.f1_score(labels, pred,
                                                                  pos_label=1))
        logger.info("Classification report:\n"
                    + classification_report(labels, pred))
        self.dict_vec = vec
        self.label_encoder = label_encoder
        self.print_model()
        logger.info("Done learning prune classifier.")

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

    def store_model(self):
        logger.info("Writing model to %s." % self.get_model_filename())
        joblib.dump([self.model, self.label_encoder,
                     self.dict_vec, self.scaler], self.get_model_filename())
        logger.info("Done.")

    def prune_candidates(self, query_candidates, key):
        remaining = []
        candidates = [key(q) for q in query_candidates]
        features = []
        for c in candidates:
            c_features = self.feature_extractor.extract_features(c)
            features.append(c_features)
        X = self.dict_vec.transform(features)
        X = self.scaler.transform(X)
        p = self.model.predict(X)
        # c = self.prune_label_encoder.inverse_transform(p)
        for candidate, predict in zip(query_candidates, p):
            if predict == 1:
                remaining.append(candidate)
        return remaining


class RelationNgramScorer(MLModel):
    """Learns a scoring based on question ngrams."""

    def __init__(self,
                 name,
                 regularization_C,
                 percentile=None):
        name += self.get_relscorer_suffix()
        MLModel.__init__(self, name, None)
        # Note: The model is lazily when needed.
        self.model = None
        self.regularization_C = regularization_C
        self.top_percentile = percentile
        self.label_encoder = None
        self.dict_vec = None
        self.scaler = None
        # The index of the correct label.
        self.correct_index = -1
        self.feature_extractor = FeatureExtractor(False,
                                                  True,
                                                  entity_features=False)

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

    def learn_model(self, train_queries):
        if self.top_percentile:
            logger.info("Collecting frequent n-gram features...")
            n_grams_dict = get_top_chi2_candidate_ngrams(train_queries,
                                                         self.feature_extractor,
                                                         percentile=self.top_percentile)
            logger.info("Collected %s n-gram features" % len(n_grams_dict))
            self.feature_extractor.ngram_dict = n_grams_dict
        features, labels = construct_examples(train_queries,
                                              self.feature_extractor)
        logger.info("#of labeled examples: %s" % len(features))
        logger.info("#labels non-zero: %s" % sum(labels))
        label_encoder = LabelEncoder()
        logger.info(features[-1])
        labels = label_encoder.fit_transform(labels)
        vec = DictVectorizer(sparse=True)
        scaler = StandardScaler(with_mean=False)
        X = vec.fit_transform(features)
        X = scaler.fit_transform(X)
        X, labels = utils.shuffle(X, labels, random_state=999)
        logger.info("#Features: %s" % len(vec.vocabulary_))
        # Perform grid search or use provided C.
        if self.regularization_C is None:
            logger.info("Performing grid search.")
            relation_scorer = SGDClassifier(loss='log', class_weight='balanced',
                                            n_iter=np.ceil(
                                                10 ** 6 // len(labels)),
                                            random_state=999)
            cv_params = [{"alpha": [10.0, 5.0, 2.0, 1.5, 1.0, 0.5,
                                    0.1, 0.01, 0.001, 0.0001]}]
            grid_search_cv = GridSearchCV(relation_scorer,
                                                      cv_params,
                                                      n_jobs=8,
                                                      verbose=1,
                                                      cv=8,
                                                      refit=True)
            grid_search_cv.fit(X, labels)
            logger.info("Best score: %.5f" % grid_search_cv.best_score_)
            logger.info("Best params: %s" % grid_search_cv.best_params_)
            self.model = grid_search_cv.best_estimator_
        else:
            logger.info("Learning relation scorer with C: %s."
                        % self.regularization_C)
            relation_scorer = SGDClassifier(loss='log', class_weight='balanced',
                                            n_iter=np.ceil(
                                                10 ** 6 // len(labels)),
                                            alpha=self.regularization_C,
                                            random_state=999)
            relation_scorer.fit(X, labels)
            logger.info("Done.")
            self.model = relation_scorer
        self.dict_vec = vec
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.correct_index = label_encoder.transform([1])[0]
        self.print_model()

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
        features = self.feature_extractor.extract_features(candidate)
        X = self.dict_vec.transform(features)
        X = self.scaler.transform(X)
        prob = self.model.predict_proba(X)
        # Prob is an array of n_examples, n_classes
        score = prob[0][self.correct_index]
        return RankScore(score)


class SimpleScoreRanker(Ranker):
    """Ranks based on a simple score of relation and entity matches."""

    def __init__(self, name, **kwargs):
        Ranker.__init__(self, name, **kwargs)

    def score(self, query_candidate):
        result_size = query_candidate.get_result_count()
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

    def rank_query_candidates(self, query_candidates, key=lambda x: x):
        """Rank query candidates by scoring and then sorting them.

        :param query_candidates:
        :return:
        """
        for qc in query_candidates:
            candidate = key(qc)
            candidate.rank_score = self.score(candidate)
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
        # Each entity match represents a matched entity.
        num_entity_matches = len(query_candidate.matched_entities)
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
            if em.entity.perfect_match or em.entity.surface_score > threshold:
                literal_entities += 1
                literal_length += len(em.entity.tokens)
            em_score = em.entity.surface_score
            em_score *= len(em.entity.tokens)
            em_token_score += em_score
            if em.entity.score > 0:
                em_popularity += math.log(em.entity.score)
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


def get_top_chi2_candidate_ngrams(queries, f_extractor, percentile):
    """Get top ngrams features according to chi2.
    """
    ngrams_dict = dict()
    features, labels = construct_examples(queries, f_extractor)
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


def construct_pair_examples(queries, f_extractor):
    """Construct training examples from candidates using pair-wise transform.

    Construct a list of examples from the given evaluated queries.
    Returns a list of features and a list of corresponding labels
    :type queries list[EvaluationQuery]
    :return:
    """
    logger.info("Extracting features from candidates.")
    labels = []
    features = []
    for query in queries:
        oracle_position = query.oracle_position
        # Only create pairs for which we "know" a correct solution
        # The oracle answer is the one with highest F1 but not necessarily
        # perfect.
        correct_cands = set()
        candidates = [x.query_candidate for x in query.eval_candidates]
        for i, candidate in enumerate(candidates):
            if i + 1 == oracle_position:
                correct_cands.add(candidate)
        if correct_cands:
            candidates = [x.query_candidate for x in query.eval_candidates]
            n_candidates = len(candidates)
            sample_size = n_candidates // 2
            if sample_size < 200:
                sample_size = min(200, n_candidates)
            sample_candidates = random.sample(candidates, sample_size)
            #sample_candidates = candidates
            for candidate in sample_candidates:
                for correct_cand in correct_cands:
                    if candidate in correct_cands:
                        continue
                    correct_cand_features = f_extractor.extract_features(correct_cand)
                    candidate_features = f_extractor.extract_features(candidate)
                    diff = feature_diff(correct_cand_features,
                                        candidate_features)
                    features.append(diff)
                    labels.append(1)
                    diff = feature_diff(candidate_features,
                                        correct_cand_features)
                    features.append(diff)
                    labels.append(0)
    return features, labels


def construct_examples(queries, f_extractor):
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
        oracle_position = query.oracle_position
        candidates = [x.query_candidate for x in query.eval_candidates]
        for i, candidate in enumerate(candidates):
            candidate_features = f_extractor.extract_features(candidate)
            features.append(candidate_features)
            if i + 1 == oracle_position:
                labels.append(1)
            else:
                labels.append(0)
    return features, labels


def construct_ngram_examples(queries, f_extractor):
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
            if query.eval_candidates[i].evaluation_result.f1 == 1.0 \
                    or i + 1 == oracle_position:
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
    keys = set(chain(features_a.keys(), features_b.keys()))
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
    Random(RANDOM_SHUFFLE).shuffle(stable_candidates)
    return stable_candidates
