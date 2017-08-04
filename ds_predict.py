"""Learn a simple logreg model on data from distant supervision.
Predict on another set of sentences only keeping those with high confidence.
Should filter out weak, noisy sentences.

"""
import logging
import util
import re
import random
import plac
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn import utils
from collections import Counter
import gzip


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def filter_examples(labels, examples, min_rel_occ=500,
                    num_classes=10, max_num_rel_examples=3000):
    """Filter examples.

    :param labels:
    :param examples:
    :param min_rel_occ: Keep only relations that occur at least that often.
    :param num_classes: Keep only the num_classes most frequent classes.
    :param max_num_rel_examples: Keep only max_num_rel_examples examples per
    relation.
    :return:
    """

    logger.info("Filtering examples")
    class_counter = Counter()
    class_counter.update(labels)
    logger.info("Most frequent classes: %s" % class_counter.most_common(3))
    # Only use the top classes by number
    # Select all classes that have at least min_rel_occ examples
    allowed_classes = set([c for c, count
                           in class_counter.most_common(num_classes)
                           if count >=min_rel_occ])
    allowed_labels, allowed_examples = [], []
    class_counter = Counter()
    num_ignored = 0
    # Remove all classes not allowed
    for label, example in zip(labels, examples):
        if label in allowed_classes:
            # Ignore the example if we have enough for that label.
            if label in class_counter and \
                    class_counter[label] >= max_num_rel_examples:
                num_ignored += 1
                continue
            class_counter.update([label])
            allowed_labels.append(label)
            allowed_examples.append(example)
    logger.info("#labels: %d, #examples: %d" % (len(allowed_classes),
                                                len(examples)))
    logger.info("Ignored %d examples because too many for that relation" % num_ignored)
    return allowed_labels, allowed_examples



def read_ds_examples(sentences_file):
    """Read the DS examples from file. Return a list of tuples
    ([relations], text).

    :param sentences_file:
    :return:
    """
    examples = []
    labels = []
    n_lines = 0
    label_set = set()
    with gzip.open(sentences_file, "rt") as f:
        for line in f:
            cols = line.strip().split('\t')
            sentence = cols[0]
            n_lines += 1
            if n_lines % 2000000 == 0:
                logger.info("Processed %d lines." % n_lines)
            rel = cols[1]
            examples.append(sentence)
            labels.append(rel)
            label_set.add(rel)
    labels, examples = utils.shuffle(labels, examples)
    logger.info("Read %d examples for %d classes." % (len(examples),
                                                      len(label_set)))
    return labels, examples


def learn_classifier(labels, examples):
    """Learn a multi-class multi-label classifier.

    :param labels:
    :param examples:
    :return:
    """
    logger.info("Training on %d examples." % len(examples))
    #logreg = OneVsRestClassifier(SGDClassifier(loss="log", n_jobs=4,
    #                                           class_weight="balanced"))
    #logreg = OneVsRestClassifier(LogisticRegression(n_jobs=4,
    #                                                solver='sag',
    #                                                class_weight="balanced"))
    #logreg = SGDClassifier(loss="log", n_jobs=4, class_weight="balanced")
    logreg = LogisticRegression(n_jobs=4, solver='sag', multi_class='ovr')
    label_encoder = LabelEncoder()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10)
    y = label_encoder.fit_transform(labels)
    X = tfidf.fit_transform(examples)
    logger.info("%d features." % X.shape[1])
    #gs = GridSearchCV(logreg, {"C": [0.1, 1.0, 10.0]},
    #                  verbose=True)
    #gs.fit(X, y)
    logreg.fit(X,y)
    #print(gs.best_params_)
    #print(gs.best_score_)
    #logreg = gs.best_estimator_
    return logreg, label_encoder, tfidf


def predict_and_write_examples(examples,
                               labels,
                               output_file, model, label_encoder, tfidf,
                               min_proba=0.8):
    """Predict labels for the examples. Only output labels if they are
    min_proba.

    :param examples:
    :param output_file:
    :param model:
    :param label_encoder:
    :param tfidf:
    :param min_proba:
    :return:
    """
    logger.info("Extracting features from %d examples." % len(examples))
    X = tfidf.transform(examples)
    logger.info("Predicting.")
    probas = model.predict_proba(X)
    predicted_labels = label_encoder.classes_
    logger.info("Writing result to %s." % output_file)
    with open(output_file, "w") as f:
        for example, label, proba in zip(examples, labels, probas):
            for p, l in zip(proba, predicted_labels):
                if l == label and p > min_proba:
                    f.write("%s\t%s\t%.2f\n" % (example, l, p))


def write_examples(labels, examples, out_file):
    with open(out_file, "w") as f:
        for l, e in zip(labels, examples):
            f.write("%s\t%s\n" % (e, l))

@plac.annotations(
    examples_file="File with ds-annotated sentences.",
    output_file="Where to write filtered sentences.",
)
def main(examples_file, output_file):
    labels, examples = read_ds_examples(examples_file)
    num_examples = len(examples)
    train_ratio = 0.9
    num_rain = int(num_examples * train_ratio)
    train_labels = labels[:num_rain]
    train_examples = examples[:num_rain]
    test_labels = labels[num_rain:]
    test_examples = examples[num_rain:]

    train_labels, train_examples = filter_examples(train_labels,
                                                   train_examples,
                                                   num_classes=1000)
    write_examples(train_labels, train_examples, "example_out")
    model, label_encoder, tfidf = learn_classifier(train_labels,
                                                   train_examples)
    predict_and_write_examples(test_examples, test_labels,
                               output_file,
                               model, label_encoder, tfidf)

if __name__ == '__main__':
    plac.call(main)
