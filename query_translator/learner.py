"""
A module for learning and evaluating ranking models.

It supports caching queries and their translation candidates to disk
in order to allow faster training and evaluation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging
import globals
from query_translator.ranker import MLModel
import scorer_globals
from query_translator.translator import QueryTranslator
from evaluation import EvaluationQuery, load_eval_queries, \
    evaluate_translator, evaluate
import os
import ranker
import cPickle as pickle
import functools
import random
from joblib import Parallel, delayed
import gc
import multiprocessing as mp
from collections import defaultdict
import translator
from sklearn.cross_validation import KFold

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
cache_directory = "data/learning_cache/"


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
                             num_processes=2):
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
    logger.info("Parallelly rescoring candidates.")
    queries = pool.map(re_rank, test_queries,
                       len(test_queries) / num_processes)
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
    for query in test_queries:
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
    params_suffix = translator.get_suffix_for_params(parameters)
    try:
        dataset_file = scorer_globals.DATASETS[dataset]
    except KeyError:
        logger.error("Unknown dataset: %s" % dataset)
        exit(1)
    filename = dataset_file.split('/')[-1]
    filename += params_suffix
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
        logger.info("Caching queries in %s." % cached_filename)
        pickle.dump(queries, open(cached_filename, 'wb'))


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
        candidate_scorer = ranker.LiteralRanker('DefaultScorer')
        candidate_scorer.parameters = parameters
        translator.set_scorer(candidate_scorer)
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


def train(scorer_name, cached):
    """Train the scorer with provided name.

    :param scorer_name:
    :param config:
    :param cached:
    :return:
    """
    try:
        scorer_obj = scorer_globals.scorers_dict[scorer_name]
    except KeyError:
        logger.error("Unknown scorer: %s" % scorer_name)
        exit(1)
    train_dataset = scorer_obj.train_dataset
    train_queries = get_evaluated_queries(train_dataset,
                                          cached,
                                          scorer_obj.get_parameters())
    logger.info("Loaded %s queries for training." % len(train_queries))
    logger.info("Training model.")
    scorer_obj.learn_model(train_queries)
    scorer_obj.store_model()
    scorer_obj.print_model()
    logger.info("Done training.")


def test(scorer_name, test_dataset, cached, avg_runs=1):
    """Evaluate the scorer on the given test dataset.

    :param scorer_name:
    :param test_dataset:
    :param config:
    :param cached:
    :return:
    """
    scorer_obj = scorer_globals.scorers_dict[scorer_name]
    # Not all rankers are MLModels
    if isinstance(scorer_obj, MLModel):
        scorer_obj.load_model()
    queries = get_evaluated_queries(test_dataset,
                                    cached,
                                    scorer_obj.get_parameters())
    result = defaultdict(int)
    n_runs = 1
    for _ in range(avg_runs):
        logger.info("Run %s of %s" % (n_runs, avg_runs))
        n_runs += 1
        res, test_queries = evaluate_scorer_parallel(queries,
                                                     scorer_obj,
                                                     num_processes=2)
        logger.info(res)
        for k, v in res._asdict().iteritems():
            result[k] += v
        gc.collect()
    for k, v in result.iteritems():
        result[k] = float(result[k]) / avg_runs
    logger.info("Average results over %s runs: " % avg_runs)
    for k in sorted(result.keys()):
        logger.info("%s: %.4f" % (k, result[k]))


def cv(scorer_name, dataset, cached, n_folds=6, avg_runs=1):
    """Report the average results across different folds.

    :param scorer_name:
    :param dataset:
    :param config:
    :param cached:
    :return:
    """
    scorer_obj = scorer_globals.scorers_dict[scorer_name]
    # Split the queries into n_folds

    queries = get_evaluated_queries(dataset,
                                    cached,
                                    scorer_obj.get_parameters())
    fold_size = len(queries) / n_folds
    logger.info("Splitting into %s folds with %s queries each." % (n_folds,
                                                                   fold_size))
    kf = KFold(len(queries), n_folds=n_folds, shuffle=True,
               random_state=999)
    result = defaultdict(int)
    n_runs = 1
    for _ in range(avg_runs):
        logger.info("Run %s of %s" % (n_runs, avg_runs))
        n_runs += 1
        num_fold = 1
        for train, test in kf:
            gc.collect()
            logger.info("Evaluating fold %s/%s" % (num_fold, n_folds))
            test_fold = [queries[i] for i in test]
            train_fold = [queries[i] for i in train]
            scorer_obj.learn_model(train_fold)
            num_fold += 1
            res, test_queries = evaluate_scorer_parallel(test_fold,
                                                         scorer_obj,
                                                         num_processes=2)
            logger.info(res)
            for k, v in res._asdict().iteritems():
                result[k] += v
            gc.collect()
    for k, v in result.iteritems():
        result[k] = float(result[k]) / (n_folds * avg_runs)
    logger.info("Average results over %s runs: " % avg_runs)
    for k in sorted(result.keys()):
        logger.info("%s: %.4f" % (k, result[k]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Learn or test a'
                                                 ' scorer model.')
    parser.add_argument('--no-cached',
                        default=False,
                        action='store_true',
                        help='Don\'t use cached data if available.')
    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use.')
    subparsers = parser.add_subparsers(help='command help')
    train_parser = subparsers.add_parser('train', help='Train a scorer.')
    train_parser.add_argument('scorer_name',
                              help='The scorer to train.')
    train_parser.set_defaults(which='train')
    test_parser = subparsers.add_parser('test', help='Test a scorer.')
    test_parser.add_argument('scorer_name',
                             help='The scorer to test.')
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
    globals.read_configuration(args.config)
    # Fix randomness.
    random.seed(999)
    use_cache = not args.no_cached
    if args.which == 'train':
        train(args.scorer_name, use_cache)
    elif args.which == 'test':
        test(args.scorer_name, args.test_dataset, use_cache,
             avg_runs=args.avg_runs)
    elif args.which == 'cv':
        cv(args.scorer_name, args.dataset, use_cache, n_folds=args.n_folds,
           avg_runs=args.avg_runs)


if __name__ == '__main__':
    # This avoids shadowing variables.
    main()
