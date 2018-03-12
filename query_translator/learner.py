"""
A module for learning and evaluating ranking models.

It supports caching queries and their translation candidates to disk
in order to allow faster training and evaluation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging
import os
import sys
import pickle
import functools
import random
import json
from joblib import Parallel, delayed
import gc
import multiprocessing as mp
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold

import config_helper
from query_translator.translator import QueryTranslator
from query_translator import ranker
import scorer_globals
from query_translator.evaluation import EvaluationQuery, load_eval_queries, \
    evaluate_translator, evaluate

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def rank_candidates(query, ranker):
    """Rerank candidates of single query.

    :rtype query: EvaluationQuery
    :param query:
    :param scorer:
    :return:
    """
    query.eval_candidates = ranker.rank_query_candidates(query.eval_candidates,
                                                         key=lambda
                                                           x: x.query_candidate)
    return query


def evaluate_scorer_parallel(test_queries, scorer_obj,
                             num_processes=1):
    """Parallel rank the candidates and evaluate the result.

    :rtype (EvaluationResult, list[EvaluationQuery])
    :param test_dataset:
    :param config:
    :param cached:
    :param scorer_obj:
    :param num_processes:
    :return:
    """
    re_rank = functools.partial(rank_candidates, ranker=scorer_obj)
    pool = mp.Pool(processes=num_processes)
    logger.info("Parallelly rescoring candidates. %r", type(test_queries[0]))
    queries = pool.map(re_rank, test_queries,
                       len(test_queries) // num_processes)
    pool.close()
    logger.info("Evaluating re-scored candidates.")
    res, queries = evaluate(queries)
    return res, queries


def evaluate_scorer(test_queries, scorer_obj):
    """Rank the candidates and evaluate the result.

    :rtype (EvaluationResult, list[EvaluationQuery])
    :param test_dataset:
    :param config:
    :param cached:
    :param scorer_obj:
    :param num_processes:
    :return:
    """
    total_queries = len(test_queries)
    for i, query in enumerate(test_queries):
        logger.info("Evaluating query %d/%d.", i + 1, total_queries)
        query.eval_candidates = scorer_obj.rank_query_candidates(
            query.eval_candidates,
            key=lambda x: x.query_candidate)
    res, queries = evaluate(test_queries)
    return res, queries


def get_cache_name_for_dataset_and_params(dataset,
                                          parameters):
    """Return the cache-filename for given dataset and parameters.

    :param dataset:
    :param parameters:
    :return:
    """
    params_suffix = parameters.get_suffix()
    try:
        dataset_file = scorer_globals.DATASETS[dataset]
    except KeyError:
        logger.error("Unknown dataset: %s" % dataset)
        exit(1)
    filename = dataset_file.split('/')[-1]
    filename += params_suffix 
    cache_directory = config_helper.config.get('model-directory')
    cached_filename = cache_directory + filename + ".cached"
    return cached_filename


def get_cached_evaluated_queries(dataset, parameters):
    """Return queries and their candidates that have been evaluated.

    For faster loading, can store the result and read from disk.
    :param dataset:
    :return:
    """
    cached_filename = get_cache_name_for_dataset_and_params(dataset,
                                                            parameters)
    if os.path.exists(cached_filename):
        logger.info("Reading cached queries from %s." % cached_filename)
        queries = pickle.load(open(cached_filename, 'rb'))
        logger.info("Read %s cached queries from disk." % len(queries))
        return queries
    else:
        return None


def cache_evaluated_queries(dataset, queries, parameters):
    """Return queries and their candidates that have been evaluated.

    For faster loading, can store the result and read from disk.
    :param dataset:
    :return:
    """
    import os
    cached_filename = get_cache_name_for_dataset_and_params(dataset,
                                                            parameters)
    # Only write if cache-file doesn't exist already.
    if not os.path.exists(cached_filename):
        try:
            logger.info("Caching queries in %s." % cached_filename)
            pickle.dump(queries, open(cached_filename, 'wb'))
        except:
            logger.error("Error during query dump", sys.exc_info()[0])


def get_evaluated_queries(dataset, cached, parameters, n_top=2000):
    """Returns evaluated queries.

    :rtype list[EvaluationQuery]
    :param dataset:
    :param config:
    :param cached:
    :param parameters:
    :param n_top:
    :return:
    """
    queries = []
    if cached:
        queries = get_cached_evaluated_queries(dataset,
                                               parameters)
    if not queries:
        # Note: we use the default scorer here, but with parameters
        # of the selected scorer.
        translator = QueryTranslator.init_from_config()
        candidate_ranker = ranker.LiteralRanker('DefaultScorer')
        candidate_ranker.parameters = parameters
        translator.set_ranker(candidate_ranker)
        queries = load_eval_queries(dataset)
        # We evaluate the queries here, so that in subsequent runs, we already
        # know which candidate is correct etc. and do not have to perform the
        # same calculations again.
        _, queries = evaluate_translator(translator,
                                         queries,
                                         n_top=n_top,
                                         ignore_invalid=False,
                                         output_result=False)
        if cached:
            cache_evaluated_queries(dataset, queries, parameters)
    return queries


def train(scorer_name, override, cached):
    """Train the scorer with provided name.

    :param scorer_name:
    :param config:
    :param cached:
    :return:
    """
    try:
        if override != {}:
            logger.info('Overrides: %s', json.dumps(override))
        scorer_obj = scorer_globals.scorers_dict[scorer_name].instance(
            override)
    except KeyError:
        logger.error("Unknown scorer: %s", scorer_name)
        exit(1)
    train_datasets = scorer_obj.train_datasets
    train_queries_all = []
    for train_dataset in train_datasets:
        train_queries = get_evaluated_queries(train_dataset,
                                              cached,
                                              scorer_obj.get_parameters(),
                                              n_top=2000)
        logger.info("Loaded %s queries for training on %s",
                    len(train_queries), train_dataset)
        train_queries_all.extend(train_queries)

    logger.info("Training model on %s queries", len(train_queries_all))
    scorer_obj.learn_model(train_queries_all)
    scorer_obj.print_model()
    scorer_obj.store_model()
    logger.info("Done training.")


def write_result_info(result, avg_runs, scorer_conf, suffix):
    """
    Writes information about the result on the logger and into a
    result info JSON file
    """
    result['runs'] = avg_runs

    logger.info('Results from %s runs', avg_runs)
    for key, value in result.items():
        logger.info('%s = %r', key, value)

    result_dir = config_helper.config.get('Learner', 'result-info-dir')
    with open(result_dir+scorer_conf.name+suffix+'_result.json',
              'w', encoding='utf8') as result_file:
        result_info = {
            'name': scorer_conf.name,
            'config': scorer_conf.config(),
            'override': scorer_conf.override(),
            'result': result}
        json.dump(result_info, result_file, sort_keys=True, indent=1)


def test(scorer_name, override, test_dataset, cached, avg_runs=1):
    """Evaluate the scorer on the given test dataset.

    :param scorer_name:
    :param test_dataset:
    :param config:
    :param cached:
    :return:
    """
    try:
        if override != {}:
            logger.info('Overrides: %s', json.dumps(override))
        scorer_conf = scorer_globals.scorers_dict[scorer_name]
        scorer_obj = scorer_conf.instance(
            override)
    except KeyError:
        logger.error("Unknown scorer: %s", scorer_name)
        exit(1)
    # Not all rankers are MLModels
    if isinstance(scorer_obj, ranker.MLModel):
        scorer_obj.load_model()
    queries = get_evaluated_queries(test_dataset,
                                    cached,
                                    scorer_obj.get_parameters(),
                                    n_top=2000)
    result = defaultdict(int)
    n_runs = 1
    for _ in range(avg_runs):
        logger.info("Run %s of %s" % (n_runs, avg_runs))
        n_runs += 1
        res, _ = evaluate_scorer(queries,
                                            scorer_obj)
        logger.info(res)
        for key, value in res._asdict().items():
            result[key] += value
        gc.collect()
    for key, value in result.items():
        result[key] = float(result[key]) / avg_runs
    write_result_info(result, avg_runs, scorer_conf, '_test')


def cv(scorer_name, override, dataset, cached, n_folds=3, avg_runs=1):
    """Report the average results across different folds.
    """
    try:
        if override != {}:
            logger.info('overrides: %s', json.dumps(override))
        scorer_conf = scorer_globals.scorers_dict[scorer_name]
        scorer_obj = scorer_conf.instance(override)
    except KeyError:
        logger.error("Unknown scorer: %s", scorer_name)
        exit(1)
    queries = get_evaluated_queries(dataset,
                                    cached,
                                    scorer_obj.get_parameters(),
                                    n_top=2000)
    fold_size = len(queries) // n_folds
    logger.info("Splitting into %s folds with %s queries each.",
                n_folds, fold_size)
    # Split the queries into n_folds
    kf = KFold(n_splits=n_folds, shuffle=True,
               random_state=999)
    result = defaultdict(int)
    n_runs = 1
    for _ in range(avg_runs):
        logger.info("Run %s of %s", n_runs, avg_runs)
        n_runs += 1
        num_fold = 1
        for train_indices, test_indices in kf.split(queries):
            gc.collect()
            logger.info("Evaluating fold %s/%s", num_fold, n_folds)
            test_fold = [queries[i] for i in test_indices]
            train_fold = [queries[i] for i in train_indices]
            scorer_obj.learn_model(train_fold)
            num_fold += 1
            res, _ = evaluate_scorer(test_fold,
                                     scorer_obj)
            logger.info(res)
            for key, value in res._asdict().items():
                result[key] += value
            gc.collect()
    for key, value in result.items():
        result[key] = float(value) / (n_folds * avg_runs)
    write_result_info(result, avg_runs, scorer_conf, '_cv')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Learn or test a'
                                                 ' scorer model.')
    parser.add_argument('--cached',
                        default=False,
                        action='store_true',
                        help='Use cached data if available.')
    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use.')
    subparsers = parser.add_subparsers(help='command help')

    train_parser = subparsers.add_parser('train', help='Train a scorer.')
    train_parser.add_argument('scorer_name',
                              help='The scorer to train.')
    train_parser.add_argument('--override', default='{}',
                              help='Override parameters of the scorer with JSON map')
    train_parser.set_defaults(which='train')

    test_parser = subparsers.add_parser('test', help='Test a scorer.')
    test_parser.add_argument('scorer_name',
                             help='The scorer to test.')
    test_parser.add_argument('--override', default='{}',
                              help='Override parameters of the scorer with JSON map')
    test_parser.add_argument('test_dataset',
                             help='The dataset on which to test the scorer.')
    test_parser.add_argument('--avg_runs',
                             type=int,
                             default=1,
                             help='Over how many runs to average.')
    test_parser.set_defaults(which='test')

    cv_parser = subparsers.add_parser('cv', help='Cross-validate a scorer.')
    cv_parser.add_argument('scorer_name',
                           help='The scorer to test.')
    cv_parser.add_argument('--override', default='{}',
                              help='Override parameters of the scorer with JSON map')
    cv_parser.add_argument('dataset',
                           help='The dataset on which to compute cv scores.')
    cv_parser.add_argument('--n_folds',
                           type=int,
                           default=6,
                           help='The number of folds.')
    cv_parser.add_argument('--avg_runs',
                           type=int,
                           default=1,
                           help='Over how many runs to average.')
    cv_parser.set_defaults(which='cv')

    args = parser.parse_args()
    # Read global config.
    config_helper.read_configuration(args.config)
    # Fix randomness.
    random.seed(999)
    np.random.seed(991)
    use_cache = args.cached
    if args.which == 'train':
        train(args.scorer_name,
              json.loads(args.override), use_cache)
    elif args.which == 'test':
        test(args.scorer_name,
             json.loads(args.override), args.test_dataset, use_cache,
             avg_runs=args.avg_runs)
    elif args.which == 'cv':
        cv(args.scorer_name,
           json.loads(args.override), args.dataset, use_cache, n_folds=args.n_folds,
           avg_runs=args.avg_runs)


if __name__ == '__main__':
    # This avoids shadowing variables.
    main()
