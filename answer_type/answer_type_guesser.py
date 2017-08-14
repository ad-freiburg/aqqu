#!/usr/bin/env python3
import argparse
import re
import logging
import sys
import numpy as np
from collections import Counter, defaultdict
from operator import itemgetter

from .freebaseize_questions import EntityMention

from query_translator.translator import Query
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
    DATE = 1
    CLASS = 2

    def __init__(self, type, target_classes=[]):
        self.type = type
        self.target_classes = target_classes

    def as_string(self):
        if self.type == AnswerType.DATE:
            return "Date"
        else:
            return ', '.join([repr(c) for c in self.target_classes])

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
            query = Query(query_str)
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
        self.target_class_date_patterns = {"in what year",
                                           "what year",
                                           "when",
                                           "since when"}
        self.clf = None
        self.vectorizer = None

    def starts_with_date_pattern(self, query):
        for p in self.target_class_date_patterns:
            if query.startswith(p):
                return True
        return False

    def identify_target(self, query):
        if self.starts_with_date_pattern(query.query_text):
            query.target_type = AnswerType(AnswerType.DATE)
        else:
            query.target_type = AnswerType(AnswerType.CLASS)


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
        self.clf = self.train_model(X, y)


    def test(self, test_file):
        featurized_queries = []
        answer_types = []
        for query, answer in gq_read(test_file):
            featurized_query = extract_features(query)
            answer_type = transform_answer(answer)

            featurized_queries.append(featurized_query)
            answer_types.append(answer_type)

        X = self.vectorizer.transform(featurized_queries)
        print("Test data shape:", X.shape)

        y = np.asarray(answer_types)
        train_error = self.clf.score(X, y)
        print("Subset accuracy on test data:", train_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--entity_types', default='data/entity_types_clean.tsv')
    parser.add_argument('--model_file', default=None)

    #parser.add_argument('savefile')
    guesser = AnswerTypeIdentifier()
    args = parser.parse_args()

    guesser.train(args.train_file)

    if args.test_file:
        guesser.test(args.test_file)

if __name__ == "__main__":
    main()
