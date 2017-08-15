#!/usr/bin/env python3
import argparse
import re
import logging
import sys
import numpy as np
import globals
from collections import Counter, defaultdict
from operator import itemgetter

from .freebaseize_questions import EntityMention

from entity_linker.entity_linker import IdentifiedEntity

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.model_selection import KFold,cross_val_score

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

mentionregex = re.compile(r'\[[^\|\]]+?\|[^\|\]]+?\|[^\|\]]+?\|[^\]]+?\]')

PAD = 'PAD'

class AnswerType:
    """
    Represents the guessed type of an answer
    """
    CLASS = 2

    def __init__(self, type, target_classes=[]):
        self.type = type
        self.target_classes = target_classes

    def as_string(self):
            return ', '.join([repr(c) for c in self.target_classes])

class DummyQuery:
    """ 
    Query class used as stand-in for query_translator.Query during training.
    This prevents a cyclic dependency
    """
    def __init__(self, text):
        self.query_tokens = None
        self.identified_entities = None

class DummyToken:
    """ 
    Token class used as stand-in for parser.Tokens during training.
    """
    def __init__(self, token):
        self.token = token

def dumb_tokenize(text):
    toks_split = text.split(' ')
    build_tok = None
    for tok in toks_split:
        if not build_tok:
            if tok[0] != '[' or tok[-1] == ']':
                yield DummyToken(tok)
            else:
                build_tok = tok
        else:
            if tok[-1] == ']':
                yield DummyToken(build_tok+tok)
                build_tok = None
            else:
                build_tok += ' '+tok


def gq_read(gq_file_path):
    with open(gq_file_path, 'rt', encoding='utf-8', errors='replace') as gq_file:
        for line in gq_file:
            query_str, answer_str = line.split('\t')
            em_answer = EntityMention.fromString(answer_str.strip())
            ie_answer = IdentifiedEntity(em_answer.tokens,
                                          em_answer.name, em_answer.mid, 0, 0,
                                          True, entity_types=em_answer.types)
            query = DummyQuery(query_str)
            query.query_tokens = []
            query.identified_entities = []
            for position, tok in enumerate(dumb_tokenize(query_str)):
                match = mentionregex.match(tok.token)
                if match:
                    em = EntityMention.fromString(match.string, position)
                    query.query_tokens.extend([DummyToken(t) for t in em.tokens])
                    ie = IdentifiedEntity(em.tokens,
                                          em.name, em.mid, 0, 0,
                                          True,
                                          entity_types=em.types)
                    query.identified_entities.append(ie)
                else:
                    query.query_tokens.append(tok)
            yield query, ie_answer

class AnswerTypeIdentifier:
    """
    A simple classe to identify the target type
    of a query, e.g. 'DATE'
    """

    def __init__(self):
        self.clf = None
        self.vectorizer = None

    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = globals.config
        answer_type_identifier = AnswerTypeIdentifier()
        model_file = config_options.get('AnswerTypeIdentifier',
                                                      'model')
        answer_type_identifier.load_model(model_file)
        return answer_type_identifier

    def load_model(self, model_file):
        logger.info("Loading answer type model")
        self.clf, self.vectorizer = joblib.load(model_file)

    def save_model(self, model_file):
        joblib.dump((self.clf, self.vectorizer), model_file)

    def identify_target(self, query):
        query.target_type = AnswerType(AnswerType.CLASS)
        query_features = self.extract_features(query)
        logger.info("Query AnswerType Features: {}".format(
            repr(query_features)))
        prediction = self.predict_best(self.vectorizer.transform(
            query_features))
        logger.info("predict: {}".format(repr(prediction)))
        query.target_type.target_classes = prediction


    def transform_answer(self, answer):
        return answer.types[0]

    def extract_features(self, query):
        features = {}
        toks = query.query_tokens
        features['tok_0'] = toks[0].token if len(toks) > 0 else PAD
        features['tok_1'] = toks[1].token if len(toks) > 1 else PAD
        features['tok_2'] = toks[2].token if len(toks) > 2 else PAD

        mentions = query.identified_entities
        features['mtype_0']  = mentions[0].types[0] if len(mentions) > 0 else PAD
        features['mtype_1']  = mentions[1].types[0] if len(mentions) > 1 else PAD
        features['mtype_2']  = mentions[2].types[0] if len(mentions) > 2 else PAD

        all_types = [t for mention in mentions for t in mention.types]
        all_types_counter = Counter(all_types)
        type_doms = all_types_counter.most_common(1)
        features['mtype_dom'] = type_doms[0][0] if len(type_doms) > 0 else PAD
        if len(all_types_counter) > 1:
            features['mtype_second'] = all_types_counter.most_common(2)[1][0]
        else:
            features['mtype_second'] = PAD
        return features

    def train_model(self, X, y):
        self.clf = SGDClassifier(loss='modified_huber')

        kf = KFold()
        for train_indices, test_indices in kf.split(X):

            self.clf.fit(X[train_indices], y[train_indices])
            train_accuracy = self.clf.score(X[test_indices], y[test_indices])
            print("Subset accuracy on fold test data:", train_accuracy)

    def train(self, train_file):
        featurized_queries = []
        answer_types = []
        for query, answer in gq_read(train_file):
            featurized_query = self.extract_features(query)
            answer_type = self.transform_answer(answer)

            featurized_queries.append(featurized_query)
            answer_types.append(answer_type)

        self.vectorizer = DictVectorizer(sparse=True)
        X = self.vectorizer.fit_transform(featurized_queries)
        print("Training data shape:", X.shape)

        y = np.asarray(answer_types)
        self.train_model(X, y)

    def predict_best(self, X, best_n=3):
        probs = self.clf.predict_proba(X)
        best_n_indices = np.argsort(probs, axis=1)[0, -best_n:]
        return self.clf.classes_[best_n_indices].tolist()


    def test_predictions(self, X, y, num=25, best_n=3):
        X_subset = X[:num]
        y_subset = y[:num]
        for i, x in enumerate(X_subset):
            prediction = self.predict_best(x, best_n)
            print('query: {}\nprediction: {}, gold: {}\n\n'.format(
                    self.vectorizer.inverse_transform(x),
                    prediction, [y[i]]
                ))


    def test(self, test_file):
        featurized_queries = []
        answer_types = []
        for query, answer in gq_read(test_file):
            featurized_query = self.extract_features(query)
            answer_type = self.transform_answer(answer)

            featurized_queries.append(featurized_query)
            answer_types.append(answer_type)

        X = self.vectorizer.transform(featurized_queries)
        print("Test data shape:", X.shape)

        y = np.asarray(answer_types)
        train_error = self.clf.score(X, y)
        print("Subset accuracy on test data:", train_error)
        print("Some predictions:")
        self.test_predictions(X, y)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--entity_types', default='data/entity_types_clean.tsv')
    parser.add_argument('--model_file', default=None)

    #parser.add_argument('savefile')
    guesser = AnswerTypeIdentifier()
    args = parser.parse_args()

    if args.load_model and args.model_file:
        guesser.load_model(args.model_file)
    else:
        guesser.train(args.train_file)

    if args.model_file and not args.load_model:
        guesser.save_model(args.model_file)


    if args.test_file:
        guesser.test(args.test_file)

if __name__ == "__main__":
    main()
