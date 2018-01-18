#!/usr/bin/env python3
import argparse
import re
import logging
import sys
import numpy as np
import config_helper
from collections import Counter, defaultdict
from operator import itemgetter

from entity_linker.entity_linker import IdentifiedEntity
from entity_linker.entity_index_rocksdb import EntityIndex

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

PAD = 'PAD'

class EntityMention:
    def __init__(self, name=None, 
                 mid='UNK', types=['UNK'], position=0):
        self.name = name
        self.mid = mid
        self.types = types
        self.tokens = name.split(' ')
        self.span = (position, position+len(self.tokens))

    def __repr__(self):
        return "[{}|{}|{}]".format(self.name,
                self.mid, ','.join(self.types)) 

    @staticmethod
    def fromString(text, entity_index, position=0, num_types=1):
        if text[0] != '[' and text[-1] != ']':
            return EntityMention(name=text)
        else:
            splits = text[1:-1].split('|')
            mid = splits[0]
            if mid == '<VALUE>':
                types = ['year']
            elif mid == 'UNK':
                types = ['UNK']
            else:
                types = entity_index.get_types_for_mid(mid, num_types)
                if len(types) < 1:
                    logger.error("Too few types in %s", text)
                    types = ['UNK']
            name = splits[1].replace('_', ' ')
            return EntityMention(name, mid, types, position)


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
    def __init__(self):
        self.tokens = []
        self.identified_entities = []


class DummyToken:
    """
    Token class used as stand-in for parser.Tokens during training.
    """
    def __init__(self, token):
        self.orth_ = token
        self.lower_ = token.lower()


def load_entity_types(entity_types_path, max_len=None):
    entity_types_map = defaultdict(lambda: ['UNK'])
    with open(entity_types_path, 'rt', encoding='utf-8') as entity_types_file:
        for line in entity_types_file:
            mid, types = line.split('\t')
            types = types.strip()
            # list[:None] is the same as list[:] see
            # https://stackoverflow.com/q/30622809
            entity_types_map[mid] = types.split(' ')[:max_len]
    return entity_types_map


def gq_read(gq_path, entity_index):
    """
    Reads a generated questions file in freebase format and returns
    generates DummyQuery objects for each line DummyQuery objects for each line
    """
    with open(gq_path, 'rt', encoding='utf-8', errors='replace') as gq_file:
        for line in gq_file:
            query_str, answer_str = line.split('\t')
            em_answer = EntityMention.fromString(answer_str.strip(),
                                                 entity_index)
            # if the answer has unknown type the question is useless
            # skip it
            if len(em_answer.types) == 1 and em_answer.types[0] == 'UNK':
                continue
            # TODO handle entity categories
            answer_tokens = [DummyToken(tok)
                             for tok in em_answer.tokens]
            ie_answer = IdentifiedEntity(answer_tokens,
                                         em_answer.name,
                                         em_answer.mid, 0, 0,
                                         True, entity_types=em_answer.types)
            query = DummyQuery()
            for position, raw_tok in enumerate(query_str.split(' ')):
                if raw_tok.startswith('[') and raw_tok.endswith(']'):
                    em = EntityMention.fromString(raw_tok, entity_index,
                                                  position)

                    query.tokens.extend([DummyToken(t) for t in em.tokens])
                    ie = IdentifiedEntity(em.tokens,
                                          em.name, em.mid, 0, 0,
                                          True,
                                          entity_types=em.types)
                    query.identified_entities.append(ie)
                else:
                    query.tokens.append(DummyToken(raw_tok))
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
        config_options = config_helper.config
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
        # TODO(schnelle): currently the training set has no count questions
        # when it has this hack should be removed
        text = query.text
        if text.startswith('in how many') or text.startswith('how many'):
            query.target_type.target_classes = [('count', 0.9)]
            query.is_count_query = True
        else:
            query_features = self.extract_features(query)
            logger.info("Query AnswerType Features: {}".format(
                repr(query_features)))
            prediction = self.predict_best(self.vectorizer.transform(
                query_features))
            query.target_type.target_classes = prediction
        logger.info("predict: {}".format(repr(query.target_type.target_classes)))


    def transform_answer(self, answer):
        return answer.types[0]

    def extract_features(self, query):
        features = {}
        toks = query.tokens
        # TODO(schnelle) make use of spacy's numerical tokens
        features['tok_0'] = toks[0].lower_ if len(toks) > 0 else PAD
        features['tok_1'] = toks[1].lower_ if len(toks) > 1 else PAD
        features['tok_2'] = toks[2].lower_ if len(toks) > 2 else PAD

        mentions = query.identified_entities
        features['mtype_0'] = mentions[0].types[0] if len(mentions) > 0 else PAD
        features['mtype_1'] = mentions[1].types[0] if len(mentions) > 1 else PAD
        features['mtype_2'] = mentions[2].types[0] if len(mentions) > 2 else PAD

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
        self.clf = SGDClassifier(loss='log')

        kf = KFold()
        for train_indices, test_indices in kf.split(X):

            self.clf.fit(X[train_indices], y[train_indices])
            train_accuracy = self.clf.score(X[test_indices], y[test_indices])
            print("Subset accuracy on fold test data:", train_accuracy)

    def train(self, train_file, entity_types_map):
        featurized_queries = []
        answer_types = []
        for query, answer in gq_read(train_file, entity_types_map):
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
        return sorted(zip(self.clf.classes_[best_n_indices].tolist(),
            probs[0, best_n_indices].tolist()), key=itemgetter(1), reverse=True)


    def test_predictions(self, X, y, num=100, best_n=3):
        X_subset = X[:num]
        for i, x in enumerate(X_subset):
            prediction = self.predict_best(x, best_n)
            print('query: {}\nprediction: {}, gold: {}\n\n'.format(
                    self.vectorizer.inverse_transform(x),
                    prediction, [y[i]]
                ))


    def test(self, test_file, entity_types_map):
        featurized_queries = []
        answer_types = []
        for query, answer in gq_read(test_file, entity_types_map):
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
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--config",
                        default="config.cfg",
                        help="The configuration file to use.")
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--modelfile', default=None)

    args = parser.parse_args()

    config_helper.read_configuration(args.config)

    entity_index = EntityIndex.init_from_config()
    guesser = AnswerTypeIdentifier()

    if args.load_model and args.modelfile:
        guesser.load_model(args.modelfile)
    else:
        guesser.train(args.train_file, entity_index)

    if args.modelfile and not args.load_model:
        guesser.save_model(args.modelfile)

    if args.test_file:
        guesser.test(args.test_file, entity_index)

if __name__ == "__main__":
    main()
