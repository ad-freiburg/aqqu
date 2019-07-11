"""
All the code for evaluation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import logging
import random
import json
import time
from collections import namedtuple

from dateutil import parser as dateparser
import urllib3
from translator_server import map_candidates
import re

logger = logging.getLogger(__name__)


def load_eval_queries(dataset):
    """Read the datasets stored in json format.

    Returns a list of queries sorted by query-id.
    :param dataset:
    :return:
    """
    import scorer_globals
    dataset_file = scorer_globals.DATASETS[dataset]
    eval_queries = []
    if dataset_file.endswith('.json'):
        eval_queries = EvaluationQuery.queries_from_json_file(
            dataset_file)
    else:
        eval_queries = EvaluationQuery.queries_from_simple_questions(
            dataset_file)
    return eval_queries



class EvaluationQuery:
    """A query from a dataaset to be evaluated / processed.

    This class serves as the structure which ground-truth queries must
    provide.
    """

    def __init__(self, q_id, utterance,
                 targets_mids, targets_names,
                 target_parses, targets_sparqls):
        self.id = q_id
        self.utterance = utterance
        # targets are mid strings or numeric/date/.. values
        # one LIST per target interpretation
        self.targets_mids = targets_mids
        self.targets_names = targets_names
        self.targets_sparqls = targets_sparqls
        # the targeted parses if available in and implemented for the dataset
        # represented using the ER(T), ERMR(T), ERMRER(T) patterns and string
        # entitiy ids details TBD
        # one parse per target interpretation
        self.target_parses = target_parses
        # When processed, the ranked list of candidates returned.
        self.eval_candidates = []
        self.oracle_position = -1
        # These are the final results for this query.
        # If the query has at least one candidate, these are identical
        # to the first candidate's results.
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.parse_score = 0.0
        self.parse_match = False
        self.oracle_f1 = 0.0
        self.oracle_parse_score = 0.0
        self.oracle_parse_match = False
        self.oracle_parse_score = 0.0
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
        self.parse_score = 0.0
        self.parse_match = False
        self.oracle_f1 = 0.0
        self.oracle_parse_score = 0.0
        self.oracle_parse_match = False
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
    def queries_from_simple_questions(filename):
        """Load evaluation queries from SimpleQuestion TSV files"""
        # SimpleQuestions uses a simple TSV format with
        # <subject_url>\t<predicate_url>\t<object_url>\n
        skip_len = len('www.freebase.com/')

        def sq_normalize(url):
            """Normalize a SimpleQuestions URLs:
            URLs here have the form: www.freebase.com/m/04whkz5 but we want
            mids of the form m.04whkz5
            """
            return url[skip_len:].replace('/', '.')

        eval_queries = []
        with open(filename, 'r', encoding='utf-8') as sq_file:
            for idn, line in enumerate(sq_file):
                subj_url, pred_url, obj_url, utterance = line.split('\t')
                subj_mid = sq_normalize(subj_url)
                obj_mid = sq_normalize(obj_url)
                pred_mid = sq_normalize(pred_url)
                mids = [obj_mid]
                eval_queries.append(
                    EvaluationQuery(idn, utterance.strip(),
                                    [mids],
                                    None,
                                    [(subj_mid, pred_mid)],
                                    None))
        return eval_queries

    @staticmethod
    def queries_from_json_file(filename):
        """Load evaluation queries from a file."""
        queries_json = json.load(open(filename, 'r', encoding='utf-8'))
        eval_queries = []
        if isinstance(queries_json, list):
            # classic format
            for query in queries_json:
                eval_queries.append(
                    EvaluationQuery(int(query['id']),
                                    query['utterance'],
                                    None,
                                    [query['result']],
                                    None,
                                    [query.get('targetOrigSparql', None)]))
        elif isinstance(queries_json, dict):
            # WebQSP format
            try:
                questions = queries_json['Questions']
            except KeyError:
                questions = []
                conversations = queries_json['Conversations']
                for c in conversations:
                    for q in c['Questions']:
                        questions.append(q)
            for idn, question in enumerate(questions):
                utterance = question['RawQuestion']
                possible_targets = []
                possible_targets_names = []
                possible_targets_sparqls = []
                for parse in question['Parses']:
                    possible_targets_sparqls.append(parse['Sparql'])
                    result_names = []
                    results = []
                    for answer in parse['Answers']:
                        atype = answer['AnswerType']
                        if atype == 'Entity':
                            result_names.append(answer['EntityName'])
                        elif atype == 'Value':
                            result_names.append(answer['AnswerArgument'])
                        # results are mid strings or numeric/date/.. values
                        results.append(answer['AnswerArgument'])
                    possible_targets_names.append(result_names)
                    possible_targets.append(results)
                eval_queries.append(
                    EvaluationQuery(idn, utterance,
                                    possible_targets,
                                    possible_targets_names,
                                    None,
                                    possible_targets_sparqls))

        return sorted(eval_queries, key=lambda x: x.id)


class EvaluationCandidate:
    """A candidate that was executed and can be evaluated."""
    def __init__(self, query_candidate,
                 executed_sparql, prediction, prediction_names):
        self.query_candidate = query_candidate
        self.executed_sparql = executed_sparql
        self.prediction = prediction
        self.prediction_names = prediction_names
        # Is set when evaluated.
        self.evaluation_result = None

    def __getstate__(self):
        """Used during pickeling"""
        d = dict(self.__dict__)
        # d['prediction'] = []
        return d


class CandidateEvaluationResult:
    """The evaluation result for a single candidate."""

    def __init__(self, precision, recall, f1, parse_score, false_positives,
                 false_negatives):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.parse_score = parse_score
        self.parse_match = parse_score > 0.99
        self.false_positives = false_positives
        self.false_negatives = false_negatives

    def __getstate__(self):
        """When pickling we don't store false negatives and positives."""
        d = dict(self.__dict__)
        d['false_positives'] = []
        d['false_negatives'] = []
        return d


def evaluate_translator(translator, queries, n_queries=None,
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
    prev_id_list = []
    gender_dict = {"male" : "", "female" : "", "unknown" : ""}
    start_time = time.time()
    if n_queries and len(queries) > n_queries:
        # Set the seed.
        random.seed(20)
        evaluation_queries = random.sample(queries, n_queries)
    else:
        evaluation_queries = queries
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!STARTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for q in evaluation_queries:
        if ignore_howmany:
            if q.utterance.lower().startswith('how many') or \
                    q.utterance.lower().startswith('in how many'):
                continue
        if ignore_invalid:
            if not q.targets_mids and not q.targets_names:
                # Changed this, because it is debatable whether that is
                # unanswerable bc the answer can be produced.
                continue
        logger.info("Translating query (id=%s) %s of %s for evaluation.",
                    q.id, n_translated_queries + 1, len(evaluation_queries))

        idx = 0
        raw_query = q.utterance
        print("query_text", raw_query)
        # print("Previous id list before: ", prev_id_list)
        prev_id_list = get_prev_id_from_gender_dict(raw_query, gender_dict)
        # print("Previous id list after: ", prev_id_list)

        try:
            parsed_query, candidates, gender = translator.translate_and_execute_query(
                q.utterance, prev_id_list,
                n_top=n_top)

            api_data = map_candidates(raw_query, parsed_query, candidates, gender)
            # print("api_data: ", api_data)
            entity_dict = get_entity_dict(api_data)

            if api_data["candidates"] != []:
                # answer = get_answer(api_data, entity_dict, 0)
                gender_dict = get_gender_dict(api_data, entity_dict, gender_dict)
        except urllib3.exceptions.MaxRetryError:
            # In some instances virtuoso just really really doesn't want to
            # give results. Completely screwing the training sucks though.  So
            # just ignore that entire query. This doesn't affect the real
            # evaluation since this is now in a separate system
            logger.warn("Failed to translate query, skipping")
            continue
        for candidate in candidates:
            query_result_rows = candidate.query_result

            result_strs = []
            result_target_mids = []
            for row in query_result_rows:
                result_target_mids.append(row[0])
                if len(row) > 1 and row[1]:
                    result_strs.append(row[1])
                else:
                    result_strs.append(row[0])
            executed_sparql = candidate.to_sparql_query(include_name=True)
            eval_candidate = EvaluationCandidate(candidate,
                                                 executed_sparql,
                                                 result_target_mids,
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
    logger.info(result)
    if output_result:
        logger.info(result)
    return result, evaluated_queries


def write_result_output(queries, output_file="eval_out.log"):
    """Write the queries' results to an output file in a standard format.

    The output contains utterance, gold result and predicted result for each
    query. This can be evaluated with a separate script.

    TODO(schnelle): this is currently broken for WebQSP

    :param queries:
    :param output_file:
    :return:
    """
    logger.info("Writing results to %s." % output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in queries:
            q_text = q.utterance

            if q.targets_names:
                result_text = json.dumps(q.targets_names[0])
            else:
                result_text = json.dumps(q.targets_mids[0])
            actual_result = []
            if q.eval_candidates:
                actual_result = q.eval_candidates[0].prediction_names
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

def compute_parse_match(candidate, parse):
    match = 0.0

    ents = {ie.entity.sparql_name()
            for ie in candidate.query_candidate.matched_entities}
    rel_names = candidate.query_candidate.get_canonical_relation_names()
    for idx, gold_part in enumerate(parse):
        # parses a subgraphs of a bipartite graph where entities are always
        # connected by relations. Thus in a subgraph description of the form
        # (E, R, M, R, E, R, T) starting with an entity every second element
        # must be a relation/entity.
        if idx % 2 == 0 and gold_part in ents:
            match += 1.0
        elif idx % 2 == 1 and rel_names[idx // 2] == gold_part:
            match += 1.0
    match /= len(parse)
    return match


def evaluate_single_candidate(candidate, eval_query):
    """Compare the prediction against the gold results for a single candidate.

    Return precision, recall, f1, false_positives, false_negatives
    :type candidate: EvaluationCandidate
    :type eval_query: EvaluationQuery
    :rtype: list[CandidateEvaluationResult]
    :return:
    """
    candidate_results = []
    # first check if the candidate matches a ground truth parse if that is
    # available
    parse_matches = []
    best_parse_match = 0.0
    if eval_query.target_parses:
        for parse in eval_query.target_parses:
            parse_matches.append(compute_parse_match(candidate, parse))
        best_parse_match = max(parse_matches)

    # we prefer to target (m)ids but some datasets only have
    # the ground truth as human readable entity names so fallback to
    # that
    gold_targets_list = eval_query.targets_mids if eval_query.targets_mids \
        else eval_query.targets_names

    gold_targets_sets = [parse_to_set(targets)
                         for targets in gold_targets_list]

    prediction_set = parse_to_set(candidate.prediction
                                  if eval_query.targets_mids
                                  else candidate.prediction_names)

    logger.debug('prediction_set: %r', prediction_set)

    for results_num, gold_targets_set in enumerate(gold_targets_sets):
        logger.debug('gold_targets_set: %r', gold_targets_set)
        true_positives = 0.0
        false_positives = []
        false_negatives = []
        # This is fast but ignores the case where entities with identical name
        # occur multiple times but in different quantities in predicted and
        # gold list. The effect overall is negligible (<0.1%), however.
        if len(candidate.prediction_names) == \
           len(gold_targets_list[results_num]) and \
           len(gold_targets_set) != len(prediction_set):
            logger.debug("Result set has different size than result list.")
        num_gold = len(gold_targets_set)
        num_predicted = len(prediction_set)
        for res in prediction_set:
            if res in gold_targets_set:
                true_positives += 1.0
                gold_targets_set.remove(res)
            else:
                false_positives.append(res)
        false_negatives.extend(gold_targets_set)
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
            if f1 > 0.99:
                logger.debug("Perfect match: %s = %s.",
                             candidate.prediction_names,
                             gold_targets_list[results_num])
        candidate_results.append(CandidateEvaluationResult(precision,
                                                           recall, f1,
                                                           best_parse_match,
                                                           false_positives,
                                                           false_negatives))
    return candidate_results


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
    correct_queries_a = {q_a.id for q_a in queries_a if q_a.f1 > 0.99}
    correct_queries_b = {q_b.id for q_b in queries_b if q_b.f1 > 0.99}
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
                                                       'parse_acc',
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
                                                       'oracle_parse_acc',
                                                       'oracle_top_2',
                                                       'oracle_top_3',
                                                       'oracle_top_5',
                                                       'oracle_top_10',
                                                       'oracle_top_100',
                                                       'avg_oracle_position',
                                                       'avg_num_candidates'])
    num_q_no_answer = 0
    num_candidates = 0
    for query in queries:
        query.reset_results()
        if query.targets_mids:
            gold_targets_list = query.targets_mids
        else:
            gold_targets_list = query.targets_names
        candidates = query.eval_candidates
        if not gold_targets_list:
            num_q_no_answer += 1
        # We have no gold answer and no candidates.
        if not gold_targets_list and not candidates:
                query.precision = 1.0
                query.recall = 1.0
                query.f1 = 1.0
                query.parse_score = 1.0
                query.parse_match = True
                query.oracle_f1 = 1.0
                query.oracle_parse_score = 1.0
                query.oracle_parse_match = True
                query.false_negatives = []
                query.false_positives = []
        # No results -> precision = recall = f1 = 0.
        # We have a gold answer but no candidates.
        if gold_targets_list and not candidates:
            query.precision = 0.0
            query.recall = 0.0
            query.f1 = 0.0
            query.parse_score = 0.0
            query.parse_match = False
            query.false_negatives = gold_targets_list
            query.false_positives = []
        # We have candidates (but maybe no gold answer).
        else:
            num_candidates += len(candidates)
            for i, prediction in enumerate(candidates):
                best_candidate_eval = prediction.evaluation_result
                # Only compute if not already computed.
                if not best_candidate_eval:
                    candidate_evals = evaluate_single_candidate(
                        prediction, query)
                    # TODO(schnelle) for WebQSP we currently simply count
                    # the best result
                    best_candidate_eval = sorted(candidate_evals,
                                                 key=lambda ev: ev.f1,
                                                 reverse=True)[0]
                    prediction.evaluation_result = best_candidate_eval
                if i == 0:
                    query.precision = best_candidate_eval.precision
                    query.recall = best_candidate_eval.recall
                    query.f1 = best_candidate_eval.f1
                    query.parse_score = best_candidate_eval.parse_score
                    query.parse_match = query.parse_score > 0.99
                    query.false_negatives = best_candidate_eval.false_negatives
                    query.false_positives = best_candidate_eval.false_positives
                if query.oracle_f1 < best_candidate_eval.f1:
                    query.oracle_f1 = best_candidate_eval.f1
                    query.oracle_parse_score = best_candidate_eval.parse_score
                    query.oracle_parse_match = best_candidate_eval.parse_score > 0.99
                    query.oracle_position = i + 1
    num_queries = len(queries)
    num_unanswered_queries = float(len([q for q in queries
                                        if not q.eval_candidates]))
    num_answered_queries = float(len(
        [q for q in queries if q.eval_candidates]))
    completely_correct = float(len([q for q in queries if q.f1 > 0.99]))
    oracle_positions = [q.oracle_position
                        for q in queries if q.oracle_position > 0]
    avg_oracle_position = sum(oracle_positions) / float(len(oracle_positions))
    oracle_top_2 = len([p for p in oracle_positions if p <= 2])
    oracle_top_3 = len([p for p in oracle_positions if p <= 3])
    oracle_top_5 = len([p for p in oracle_positions if p <= 5])
    oracle_top_10 = len([p for p in oracle_positions if p <= 10])
    oracle_top_100 = len([p for p in oracle_positions if p <= 100])
    perfect_with_oracle = len([q for q in queries if q.oracle_f1 > 0.99])
    oracle_accuracy = perfect_with_oracle / num_queries
    oracle_top_2f = oracle_top_2 / num_queries
    oracle_top_3f = oracle_top_3 / num_queries
    oracle_top_5f = oracle_top_5 / num_queries
    oracle_top_10f = oracle_top_10 / num_queries
    oracle_top_100f = oracle_top_100 / num_queries

    # Simple averages.
    average_f1 = sum([q.f1 for q in queries]) / num_queries
    oracle_average_f1 = sum([q.oracle_f1 for q in queries]) / num_queries
    average_recall = sum([q.recall for q in queries]) / num_queries
    average_precision = sum([q.precision for q in queries]) / num_queries

    # Parse match accuracy
    parse_accuracy = len([1 for q in queries if q.parse_match]) / num_queries
    oracle_parse_accuracy = len(
        [1 for q in queries if q.oracle_parse_match]) / num_queries

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
                                      parse_accuracy,
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
                                      oracle_parse_accuracy,
                                      oracle_top_2f,
                                      oracle_top_3f,
                                      oracle_top_5f,
                                      oracle_top_10f,
                                      oracle_top_100f,
                                      avg_oracle_position,
                                      avg_num_candidates)
    if output_file:
        write_result_output(queries, output_file)
    return overall_result, queries


"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

def get_prev_id_from_gender_dict(question, gender_dict):

    male = [" he ", " his ", " him "]
    additional_male = [" he", " he?", " his", " his?", " him", " him?"]
    for am in additional_male:
        if question.endswith(am):
            male.append(am)

    female = [" she ", " her "]
    additional_female = [" she", " she?", " her", " her?"]
    for af in additional_female:
        if question.endswith(af):
            female.append(af)

    neutral = [" it ", " there ", " its "]
    additional_neutral = [" it", " there", " its", " it?", " there?", " its?"]
    for an in additional_neutral:
        if question.endswith(an):
            neutral.append(an)

    all_gender = [" they ", " their ", " thems "]
    additional_all_gender = [" they", " their", " thems", " they?", " their?", " thems?"]
    for aag in additional_all_gender:
        if question.endswith(aag):
            all_gender.append(aag)

    prev_id_list = []
    print("*************************************")
    # print("Gender dict: ", gender_dict)
    if any(x in question for x in female):
        # prev_id = request.cookies.get('prev_id_female')
        prev_id = gender_dict["female"]
        # now prev_id is a list
        if prev_id != "" and prev_id != None:
            prev_id_list = prev_id.split(";")
    elif any(x in question for x in male):
        prev_id = gender_dict["male"]
        # now prev_id is a list
        if prev_id != "" and prev_id != None:
            prev_id_list = prev_id.split(";")
    elif any(x in question for x in neutral):
        prev_id = gender_dict["unknown"]
        # now prev_id is a list
        if prev_id != "" and prev_id != None:
            prev_id_list = prev_id.split(";")
    elif any(x in question for x in all_gender):
        prev_id_list = []
        prev_id1 = gender_dict["female"]
        prev_id2 = gender_dict["male"]
        prev_id3 = gender_dict["unknown"]
        for pid in [prev_id1, prev_id2, prev_id3]:
            if pid != "":
                prev_id_list.append(pid)
    else:
        prev_id = ""
        prev_id_list = []
    """ """
    pronoun = [" he", " she", " his", " her", " him", " it", " they", " their", " there"]
    if all(x not in question for x in pronoun):
        prev_id = ""
        prev_id_list = []

    return prev_id_list

def get_entity_dict(api_data):

    """ The function filters the identified entities from answer API.
    It returns the entity dictionary, where entity id is a key
    and entity name is a value."""

    entity_dict = {}
    identified_entities = api_data["parsed_query"]["identified_entities"]
    for e in identified_entities:
        entity_dict[e["entity"]["mid"]] = e["entity"]["name"]

    return entity_dict

def get_identified_entities(api_data):

    # current_id is a set with all identified entities
    # that came up in candidates answers
    """ There will alsways be an mid for entity matches!"""

    current_id = set()
    for ans in api_data["candidates"]:
        for pid in ans["entity_matches"]:
            current_id.add(pid["mid"])
    return current_id

def get_prev_id_to_save(api_data, entity_dict):

    """ The function adds enttiy names to the prev_id
    to store it in cookies together.
    prev_id is a string with id1,name1;id2,name;..."""
    current_id = get_identified_entities(api_data)
    prev_id = ""
    for ci in current_id:
        prev_obj_name = entity_dict[ci]
        prev_obj_name = prev_obj_name.replace('PREV:', '')
        prev_id = prev_id + ";" + ci + "," + prev_obj_name
    prev_id = prev_id[1:]
    return prev_id

def get_answer_entities(api_data):

    """ Add answers entities, not sure about it."""

    answ_entity_set = set()
    first_answer = api_data["candidates"][0]["answers"]
    for fa in first_answer:
        try:
            answ_entity_set.add(fa["mid"] + "," + fa["name"])
        except KeyError:
            pass
    return answ_entity_set

def add_answer_entities(api_data, prev_id):

    """ Add the entities, that are identified in answers candidates
    to identified entites to store them together."""
    # get entities, identified in answers
    answ_entity_set = get_answer_entities(api_data)
    for aes in answ_entity_set:
        prev_id = prev_id + ";" + aes
    return prev_id

def get_prev_id_dict(api_data, prev_id):

    """ Make a dictionary out of prev_id string.""" 

    # other identified entities
    prev_id = add_answer_entities(api_data, prev_id)
    prev_id_list = prev_id.split(";")

    prev_id_dict = {}
    for pil in prev_id_list:
        ent = pil.split(",")
        try:
            prev_id_dict[ent[0]] = ent[1]
        except IndexError:
            print("No entity id or entity name.")
    return prev_id_dict

def get_gender_dict(api_data, entity_dict, gender_dict):

    """ Getting gender dictionary where keys are: male, female and unknown,
    values are id1,name1;id2,name2;..."""
    prev_id = get_prev_id_to_save(api_data, entity_dict)
    prev_id_dict = get_prev_id_dict(api_data, prev_id)
    gender_dict = {"male" : "", "female" : "", "unknown" : ""}
    for e in prev_id_dict.keys():
        gender = "unknown"
        parsed_query_obj = prev_id_dict[e]
        try:
            gender = api_data["gender"][e]
        except KeyError:
            gender = "unknown"
        if gender_dict[gender] == "":
            gender_dict[gender] = e + "," + prev_id_dict[e]
        else:
            gender_dict[gender] = gender_dict[gender] + ";" + e + "," + prev_id_dict[e]
    return gender_dict

def get_answer(api_data, entity_dict, idx):

    # get all the identified entities id of an answer
    answer_entity_id = []
    for aei in api_data["candidates"][idx]["entity_matches"]:
        answer_entity_id.append(aei["mid"])

    parsed_query_obj_name = ""
    for aei in answer_entity_id:
        if entity_dict[aei][:5] == "PREV:":
            obj_name = entity_dict[aei][5:]
        else:
            obj_name = entity_dict[aei]
        parsed_query_obj_name = parsed_query_obj_name + ", " + obj_name

    # Cut the firs ", "
    parsed_query_obj_name = parsed_query_obj_name[2:]
    if parsed_query_obj_name is '' or parsed_query_obj_name is None:
        """"""
        parsed_query_obj_name = api_data["parsed_query"]["identified_entities"][0]["entity"]["name"]
        answer_entity_id = [api_data["parsed_query"]["identified_entities"][0]["entity"]["mid"]]

    # type of question, relation that helped to find an answer
    parsed_query_type_whole = api_data["candidates"][idx]["relation_matches"]\
                                      [0]["name"]
    # Take the last part of a relation
    parsed_query_type = parsed_query_type_whole.split(".")[-1]
    # delete all non characters and non numbers
    parsed_query_type = re.sub('[^0-9a-zA-Z]+', ' ', parsed_query_type)
    # print("api_data[candidates][idx][answers]: ", api_data["candidates"][idx]["answers"])
    answer = api_data["candidates"][idx]["answers"][0]["name"]
    # if more than one answer
    for i in range(1, len(api_data["candidates"][idx]["answers"])):
        answer = answer + ", " + api_data["candidates"][idx]["answers"][i]["name"]

    return parsed_query_obj_name + ", " + parsed_query_type + ": " + str(answer)

"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
