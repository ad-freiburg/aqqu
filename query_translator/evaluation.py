"""
All the code for evaluation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from collections import namedtuple
from dateutil import parser as dateparser
import logging
import random
import json
import time

logger = logging.getLogger(__name__)


def load_eval_queries(dataset):
    """Read the datasets stored in json format.

    Returns a list of queries sorted by query-id.
    :param dataset:
    :return:
    """
    import scorer_globals
    dataset_file = scorer_globals.DATASETS[dataset]
    return EvaluationQuery.queries_from_json_file(dataset_file)


class EvaluationQuery(object):
    """A query from a dataaset to be evaluated / processed.

    This class serves as the structure which ground-truth queries must
    provide.
    """

    def __init__(self, q_id, utterance, target_result, target_sparql):
        self.id = q_id
        self.utterance = utterance
        self.target_result = target_result
        self.target_sparql = target_sparql
        # When processed, the ranked list of candidates returned.
        # TODO(Elmar): RENAME THIS!!!! VERY CONFUSING!!!
        # TODO(schnelle): Indeed I'm quite confused
        self.eval_candidates = []
        self.oracle_position = -1
        # These are the final results for this query.
        # If the query has at least one candidate, these are identical
        # to the first candidate's results.
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.oracle_f1 = 0.0
        self.false_negatives = []
        self.false_positives = []

    def reset_results(self):
        """If a query is used several times for evaluation we reset it.

        :return:
        """
        self.oracle_position = -1
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.oracle_f1 = 0.0
        self.false_negatives = []
        self.false_positives = []

    def __getstate__(self):
        """When pickling we don't store false negatives and positives.
        :return:
        """
        d = dict(self.__dict__)
        d['false_positives'] = []
        d['false_negatives'] = []
        return d

    @staticmethod
    def queries_from_json_file(filename):
        """Load evaluation queries from a file."""
        def object_decoder(q):
            return EvaluationQuery(int(q['id']),
                                   q['utterance'],
                                   q['result'],
                                   q.get('targetOrigSparql', None))
        eval_queries = json.load(open(filename, 'r', encoding = 'utf-8'),
                                 object_hook=object_decoder)
        return sorted(eval_queries, key=lambda x: x.id)


class EvaluationCandidate(object):
    """A candidate that was executed and can be evaluated."""
    def __init__(self, query_candidate, executed_sparql, prediction):
        self.query_candidate = query_candidate
        self.executed_sparql = executed_sparql
        self.prediction = prediction
        # Is set when evaluated.
        self.evaluation_result = None

    def __getstate__(self):
        """When pickling we don't store the actual result."""
        d = dict(self.__dict__)
        # d['prediction'] = []
        return d


class CandidateEvaluationResult(object):
    """The evaluation result for a single candidate."""

    def __init__(self, precision, recall, f1, false_positives,
                 false_negatives):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.is_correct = True if f1 == 1.0 else False
        self.false_positives = false_positives
        self.false_negatives = false_negatives

    def __getstate__(self):
        """When pickling we don't store false negatives and positives."""
        d = dict(self.__dict__)
        d['false_positives'] = []
        d['false_negatives'] = []
        return d


def evaluate_translator(translator, queries, n_queries=9999,
                        ignore_howmany=False, ignore_invalid=False,
                        n_top=1000, output_result=True):
    """Evaluate the translator on the provided queries.

    Returns a result object as defined in the evaluation
    module and the list of queries in the form of dictionaries holding
    evaluation information.
    :type queries: list[EvaluationQuery]
    :type translator: query_translator.translator.QueryTranslator
    :rtype: (EvaluationResult, list[EvaluationQuery])
    :param n_queries:
    :param ignore_howmany:
    :param ignore_invalid:
    :return:
    """
    translated_queries = []
    n_translated_queries = 0
    start_time = time.time()
    if len(queries) > n_queries:
        # Set the seed.
        random.seed(20)
        evaluation_queries = random.sample(queries, n_queries)
    else:
        evaluation_queries = queries
    for q in evaluation_queries:
        if ignore_howmany:
            if q.utterance.lower().startswith('how many') or \
                    q.utterance.lower().startswith('in how many'):
                continue
        if ignore_invalid:
            if len(q.target_result) == 0:
                # Changed this, because it is debatable whether that is
                # unanswerable bc the answer can be produced.
                # or len(q.target_result) == 1 and q.target_result[0] == '0':
                continue
        logger.info("Translating query (id=%s) %s of %s for evaluation." %
                    (q.id, n_translated_queries + 1, len(evaluation_queries)))
        results = translator.translate_and_execute_query(q.utterance,
                                                         n_top=n_top)
        for result in results:
            candidate = result.query_candidate
            query_result_rows = result.query_result_rows
            # Only want the readable name.
            result_strs = []
            for r in query_result_rows:
                if len(r) > 1 and r[1]:
                    result_strs.append(r[1])
                else:
                    result_strs.append(r[0])
            executed_sparql = candidate.to_sparql_query(include_name=True)
            eval_candidate = EvaluationCandidate(candidate,
                                                 executed_sparql,
                                                 result_strs)
            q.eval_candidates.append(eval_candidate)
        translated_queries.append(q)
        n_translated_queries += 1
    duration = (time.time() - start_time) * 1000
    logger.info("Total translation time for %s "
                "questions: %s ms" % (n_translated_queries,
                                      duration))
    logger.info("Translation per query: %s ms" %
                (duration / n_translated_queries))
    logger.info("Computing evaluation statistics.")
    result, evaluated_queries = evaluate(translated_queries)
    logger.info("Finished computing evaluation statistics.")
    if output_result:
        logger.info(result)
    return result, evaluated_queries


def write_result_output(queries, output_file="eval_out.log"):
    """Write the queries' results to an output file in a standard format.

    The output contains utterance, gold result and predicted result for each
    query. This can be evaluated with a separate script.
    :param queries:
    :param output_file:
    :return:
    """
    logger.info("Writing results to %s." % output_file)
    with open(output_file, 'w') as f:
        for q in queries:
            q_text = q.utterance
            result_text = json.dumps(q.target_result)
            actual_result = []
            if q.eval_candidates:
                actual_result = q.eval_candidates[0].prediction
            actual_result_text = json.dumps(actual_result)
            f.write("%s\t%s\t%s\n" % (q_text, result_text,
                                      actual_result_text))


def parse_date(date_string):
    """Try to parse the string into a date.

    Returns a tuple of year, month, day. Note that some other
    systems only compare against year so this is even more strict.
    :param date_string:
    :return:
    """
    try:
        date = dateparser.parse(date_string)
        return date.year, date.month, date.day
    # We can't parse into date.
    except:
        return None


def parse_float(float_string):
    """Try to parse the string into a float.

    :param float_string:
    :return:
    """
    try:
        return float(float_string)
    except ValueError:
        return None


def parse_to_set(result_list):
    """Transform result list into a set of values.

    :param result_list:
    :return:
    """
    result_set = set()
    for r in result_list:
        r_float = parse_float(r)
        if r_float:
            result_set.add(r_float)
            continue
        r_date = parse_date(r)
        if r_date:
            result_set.add(r_date)
            continue
        result_set.add(r)
    return result_set


def evaluate_single_candidate(candidate, eval_query):
    """Compare the prediction against the gold results for a single candidate.

    Return precision, recall, f1, false_positives, false_negatives
    :type candidate: EvaluationCandidate
    :type eval_query: EvaluationQuery
    :rtype: CandidateEvaluationResult
    :return:
    """
    true_positives = 0.0
    false_positives = []
    false_negatives = []
    gold_result_set = parse_to_set(eval_query.target_result)
    prediction_set = parse_to_set(candidate.prediction)
    # This is fast but ignores the case where entities with identical name
    # occur multiple times but in different quantities in predicted and
    # gold list. The effect overall is negligible (<0.1%), however.
    if len(candidate.prediction) == len(eval_query.target_result) and \
               len(gold_result_set) != len(prediction_set):
        logger.debug("Result set has different size than result list.")
    num_gold = len(gold_result_set)
    num_predicted = len(prediction_set)
    for res in prediction_set:
        if res in gold_result_set:
            true_positives += 1.0
            gold_result_set.remove(res)
        else:
            false_positives.append(res)
    false_negatives.extend(gold_result_set)
    if num_gold == 0:
        if num_predicted == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        if num_predicted == 0:
            precision = 0.0
        else:
            precision = true_positives / float(num_predicted)
        recall = true_positives / float(num_gold)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        if f1 == 1.0:
            logger.debug("Perfect match: %s = %s." %
                         (candidate.prediction, eval_query.target_result))
    return CandidateEvaluationResult(precision, recall, f1,
                                     false_positives, false_negatives)


def compare_evaluation_runs(queries_a, queries_b):
    """Compare queries of two evaluation runs.

    Returns two lists sorted by id: queries correct in a but incorrect in b
    and the other way around
    :type queries_a: list[EvaluationQuery]
    :type queries_b: list[EvaluationQuery]
    :rtype: (list[EvaluationQuery], list[EvaluationQuery])
    :param queries_a:
    :param queries_b:
    :return:
    """
    if len(queries_a) != len(queries_b):
        logger.error("Number of queries of different runs is different.")
        raise RuntimeError
    correct_queries_a = {q_a.id for q_a in queries_a if q_a.f1 == 1.0}
    correct_queries_b = {q_b.id for q_b in queries_b if q_b.f1 == 1.0}
    # Create maps from id to query dictionary.
    query_map_a = {q_a.id: q_a for q_a in queries_a}
    query_map_b = {q_b.id: q_b for q_b in queries_b}
    # A\B -> correct in A, but incorrect in B
    a_not_b = [query_map_b[q_id]
               for q_id in (correct_queries_a - correct_queries_b)]
    # B\A -> correct in B, but incorrect in A
    b_not_a = [query_map_a[q_id]
               for q_id in (correct_queries_b - correct_queries_a)]
    # Create sorted lists.
    a_not_b = sorted(a_not_b, key=lambda x: x.id)
    b_not_a = sorted(b_not_a, key=lambda x: x.id)
    return a_not_b, b_not_a


def evaluate(queries, output_file="eval_out.log"):
    """Evaluates the queries.

    Returns a tuple of EvaluationResult and the list of queries. Each
    query is enhanced with information on false_positives
    and false_negatives.
    :type queries: list[EvaluationQuery]
    :rtype: (EvaluationResult, list[EvaluationQuery])
    :param queries:
    :return:
    """

    EvaluationResult = namedtuple('EvaluationResult', ['avg_precision',
                                                       'avg_recall',
                                                       'avg_f1',
                                                       'macro_f1',
                                                       'macro_f1_xao',
                                                       'avg_precision_xao',
                                                       'precision_kw',
                                                       'recall_kw',
                                                       'f1_kw',
                                                       'num_questions',
                                                       'num_questions_no_answer',
                                                       'accuracy',
                                                       'oracle_accuracy',
                                                       'oracle_avg_f1',
                                                       'oracle_top_2',
                                                       'oracle_top_3',
                                                       'oracle_top_5',
                                                       'oracle_top_10',
                                                       'oracle_top_100',
                                                       'avg_oracle_position',
                                                       'avg_num_candidates'])
    num_q_no_answer = 0
    num_candidates = 0
    for q in queries:
        q.reset_results()
        gold_results = q.target_result
        candidates = q.eval_candidates
        if len(gold_results) == 0:
            num_q_no_answer += 1
        # We have no gold answer and no candidates.
        if len(gold_results) == 0 and len(candidates) == 0:
                q.precision = 1.0
                q.recall = 1.0
                q.f1 = 1.0
                q.oracle_f1 = 1.0
                q.false_negatives = []
                q.false_positives = []
        # No results -> precision = recall = f1 = 0.
        # We have a gold answer but no candidates.
        if len(gold_results) > 0 and len(candidates) == 0:
            q.precision = 0.0
            q.recall = 0.0
            q.f1 = 0.0
            q.false_negatives = gold_results
            q.false_positives = []
        # We have candidates (but maybe no gold answer).
        else:
            num_candidates += len(candidates)
            for i, prediction in enumerate(candidates):
                candidate_eval = prediction.evaluation_result
                # Only compute if not already computed.
                if not candidate_eval:
                    candidate_eval = evaluate_single_candidate(prediction, q)
                    prediction.evaluation_result = candidate_eval
                if i == 0:
                    q.precision = candidate_eval.precision
                    q.recall = candidate_eval.recall
                    q.f1 = candidate_eval.f1
                    q.false_negatives = candidate_eval.false_negatives
                    q.false_positives = candidate_eval.false_positives
                if q.oracle_f1 < candidate_eval.f1:
                    q.oracle_f1 = candidate_eval.f1
                    q.oracle_position = i + 1
    num_queries = len(queries)
    num_unanswered_queries = float(len([q for q in queries
                                        if not q.eval_candidates]))
    num_answered_queries = float(len([q for q in queries if q.eval_candidates]))
    completely_correct = float(len([q for q in queries if q.f1 == 1.0]))
    oracle_positions = [q.oracle_position
                        for q in queries if q.oracle_position > 0]
    avg_oracle_position = sum(oracle_positions) / float(len(oracle_positions))
    oracle_top_2 = len([p for p in oracle_positions if p <= 2])
    oracle_top_3 = len([p for p in oracle_positions if p <= 3])
    oracle_top_5 = len([p for p in oracle_positions if p <= 5])
    oracle_top_10 = len([p for p in oracle_positions if p <= 10])
    oracle_top_100 = len([p for p in oracle_positions if p <= 100])
    perfect_with_oracle = len([q for q in queries if q.oracle_f1 == 1.0])
    oracle_accuracy = float(perfect_with_oracle) / num_queries
    oracle_top_2 = float(oracle_top_2) / num_queries
    oracle_top_3 = float(oracle_top_3) / num_queries
    oracle_top_5 = float(oracle_top_5) / num_queries
    oracle_top_10 = float(oracle_top_10) / num_queries
    oracle_top_100 = float(oracle_top_100) / num_queries

    # Simple averages.
    average_f1 = sum([q.f1 for q in queries]) / num_queries
    oracle_average_f1 = sum([q.oracle_f1 for q in queries]) / num_queries
    average_recall = sum([q.recall for q in queries]) / num_queries
    average_precision = sum([q.precision for q in queries]) / num_queries

    # Macro F1 that considers all queries
    macro_f1 = 0.0
    if average_precision + average_recall > 0:
        macro_f1 = (2 * average_precision * average_recall
                    / (average_precision + average_recall))

    # Exact accuracy.
    accuracy = float(completely_correct) / num_queries

    # Xao et al. ignore unanswered queries when computing the precision
    # We previously set the precision of those to 0.0, so only adjust
    # denominator
    average_precision_xao = (sum([q.precision for q in queries]) /
                             (num_queries - num_unanswered_queries))
    macro_f1_xao = 0
    if average_precision_xao + average_recall > 0:
        macro_f1_xao = (2 * average_precision_xao * average_recall /
                        (average_precision_xao + average_recall))

    # Scores like Kwiatowski et al.:
    # Precision: the percentage of produced queries with correct answers
    # Recall: the percent of total questions answered correctly
    precision_kw = completely_correct / num_answered_queries
    recall_kw = completely_correct / num_queries
    f1_kw = 0.0
    if precision_kw + recall_kw > 0:
        f1_kw = 2 * precision_kw * recall_kw / (precision_kw + recall_kw)

    avg_num_candidates = float(num_candidates) / num_queries

    overall_result = EvaluationResult(average_precision,
                                      average_recall,
                                      average_f1,
                                      macro_f1,
                                      macro_f1_xao,
                                      average_precision_xao,
                                      precision_kw,
                                      recall_kw,
                                      f1_kw,
                                      num_queries,
                                      num_q_no_answer,
                                      accuracy,
                                      oracle_accuracy,
                                      oracle_average_f1,
                                      oracle_top_2,
                                      oracle_top_3,
                                      oracle_top_5,
                                      oracle_top_10,
                                      oracle_top_100,
                                      avg_oracle_position,
                                      avg_num_candidates)
    if output_file:
        write_result_output(queries, output_file)
    return overall_result, queries
