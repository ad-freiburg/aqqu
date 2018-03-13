"""
Provides a REST based translation interface.

Copyright 2017, University of Freiburg.

Niklas Schnelle <schnelle@cs.uni-freiburg.de>

"""
import logging
from typing import List, Union, Iterable, Dict, Any
import json
import sys
import config_helper
import scorer_globals
import flask
import spacy
from query_translator.query_candidate import RelationMatch,\
    QueryCandidate, QueryCandidateNode
from query_translator.translator import QueryTranslator, Query
import entity_linker.entity_linker as entity_linker


logging.basicConfig(format="%(asctime)s : %(levelname)s "
                           ": %(module)s : %(message)s",
                    level=logging.DEBUG)

LOG = logging.getLogger(__name__)


APP = flask.Flask(__name__)

class ClassNameJSONEncoder(json.JSONEncoder):
    """
    Allows JSON serializing classes as their name
    """
    def default(self, obj):
        return obj.__class__.__name__

APP.json_encoder = ClassNameJSONEncoder


def map_results_list(results: Iterable[List[Union[str, int]]]) -> List[dict]:
    """
    Maps the result rows of the SPARQL backend into JSON compatible lists
    """
    query_results = []
    for result in results:
        if len(result) > 1:
            query_results.append({'name': result[1], 'mid': result[0]})
        else:
            query_results.append({'name': result[0]})
    return query_results


def map_query_doc(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """
    Maps a spaCy Doc into a JSON compatible list
    """
    return [{'orth': tok.orth_, 'tag': tok.tag_,
             'lemma': tok.lemma_, 'offset': tok.idx} for tok in doc]


def map_entity(entity: entity_linker.Entity) -> dict:
    """
    Maps an entity_linker.Entity to a JSON compatible dict
    """
    return {'name': entity.name,
            'mid': entity.sparql_name()}


def map_identified_entity(entity: entity_linker.IdentifiedEntity)\
        -> Dict[str, Any]:
    """
    Maps an IdentifiedEntity into a JSON compatible dict
    """
    mapped_entity = {
        'token_positions': [tok.i for tok in entity.tokens],
        'surface_score': entity.surface_score,
        'score': entity.score,
        'entity': map_entity(entity.entity),
        'perfect_match': entity.perfect_match,
        'text_match': entity.text_match,
        'types': entity.types
    }
    return mapped_entity


def map_query(query: Query) -> Dict[str, Any]:
    """
    Maps a Query into a JSON compatible dict
    """
    tokens = map_query_doc(query.tokens)
    content_token_positions = [tok.i for tok in query.content_tokens]

    identified_entities = [map_identified_entity(entity)
                           for entity in query.identified_entities]
    query_map = {
        'target_type': query.target_type.as_string(),
        'tokens': tokens,
        'content_token_positions': content_token_positions,
        'identified_entities': identified_entities,
        'is_count': query.is_count_query
    }
    return query_map


def map_relation_matches(rel_matches: Iterable[RelationMatch])\
                                        -> List[Dict[str, Any]]:
    """
    Maps RelationMatches to a list of JSON compatible dicts
    """
    results = []
    for rel_match in rel_matches:
        rel_match_dict = {}
        if rel_match.name_match:
            name_match = rel_match.name_match
            token_name_dicts = [{'token_position': tok.i, 'name': name}
                                for tok, name in name_match.token_names]
            name_match_dict = {'token_names': token_name_dicts}
            rel_match_dict['name_match'] = name_match_dict
        if rel_match.derivation_match:
            derivation_match = rel_match.derivation_match
            token_name_dicts = [{'token_position': tok.i, 'name': name}
                                for tok, name in derivation_match.token_names]
            derivation_match_dict = {'token_names': token_name_dicts}
            rel_match_dict['derivation_match'] = derivation_match_dict
        if rel_match.words_match:
            words_match = rel_match.words_match
            token_name_dicts = [{'token_position': tok.i, 'score': score}
                                for tok, score in words_match.token_scores]
            words_match_dict = {'token_scores': token_name_dicts}
            rel_match_dict['words_match'] = words_match_dict
        if rel_match.name_weak_match:
            name_weak_match = rel_match.name_weak_match
            token_name_score_dicts = [{'token_position': tok.i, 'name': name,
                                       'score': score}
                                      for tok, name, score
                                      in name_weak_match.token_name_scores]
            name_weak_match_dict = {'token_name_scores':
                                    token_name_score_dicts}
            rel_match_dict['name_weak_match'] = name_weak_match_dict
        if rel_match.count_match:
            count_match = rel_match.count_match
            count_match_dict = {'count': count_match.count}
            rel_match_dict['count_match'] = count_match_dict

        relation_name = rel_match.relation
        if isinstance(rel_match.relation, tuple):
            relation_name = ' -> '.join(rel_match.relation)
        rel_match_dict['name'] = relation_name

        token_positions = [tok.i for tok in rel_match.tokens]
        rel_match_dict['token_positions'] = token_positions
        results.append(rel_match_dict)

    return results


def map_entity_matches(ent_matches: Iterable[entity_linker.IdentifiedEntity])\
                        -> List[Dict[str, Any]]:
    """
    Maps IdentifiedEntities to a list of JSON compatible dicts
    """
    return [{'mid': em.entity.sparql_name()} for em in ent_matches]


def map_query_graph(node: QueryCandidateNode, visited: set)\
                                          -> Dict[str, Any]:
    """
    Maps the query graph into a JSON compatible format starting from the
    root_node.

    NOTE: This assumes that all nodes are reachable via out_relations from the
    root_node
    """
    if node in visited:
        return None

    out_relations = []  # type: List[Dict[str, str]]
    visited.add(node)
    for rel in node.out_relations:
        rel_dict = {'name': rel.name, 'target_node': None}
        if rel.has_target() and rel.target_node not in visited:
            rel_dict['target_node'] = map_query_graph(rel.target_node, visited)
        out_relations.append(rel_dict)

    mid = None
    if node.entity_match:
        mid = node.entity_match.entity.sparql_name()
    return {'mid': mid,
            'out_relations': out_relations}


def map_candidate(candidate: QueryCandidate) -> Dict[str, Any]:
    """
    Turns a QueryCandidate into a result dict suitable for JSON encoding
    """
    answers = map_results_list(candidate.query_result)
    relation_matches = map_relation_matches(candidate.matched_relations)
    entity_matches = map_entity_matches(candidate.matched_entities)
    visited = set()  # type: Set[Any]
    root_node = map_query_graph(candidate.root_node, visited)

    return {
        'sparql': candidate.to_sparql_query(),
        'rank_score': candidate.rank_score,
        'matches_answer_type': candidate.matches_answer_type,
        'features': candidate.feature_dict,
        'relation_matches': relation_matches,
        'entity_matches': entity_matches,
        'pattern': candidate.pattern,
        'root_node': root_node,
        'answers': answers,
        }


def map_candidates(raw_query: str, parsed_query: Query,
                   candidates: Iterable[QueryCandidate])\
                                             -> Dict[str, Any]:
    """
    Turns the final translations which are lists of namedtuple with
    query_candidate and query_result into a result map suitable for JSON
    encoding
    """
    candidate_dicts = []  # type: List[dict]
    for candidate in candidates:
        candidate_dicts.append(map_candidate(candidate))

    return {'raw_query': raw_query,
            'parsed_query': map_query(parsed_query),
            'candidates': candidate_dicts}

def main() -> None:
    """
    Entry point into the program
    """
    import argparse
    parser = argparse.ArgumentParser(description="REST api based translation.")
    parser.add_argument("ranker_name",
                        default="WQ_Ranker",
                        help="The ranker to use.")
    parser.add_argument('--override', default='{}',
                        help='Override parameters of the ranker with JSON map')
    parser.add_argument("--config",
                        default="config.cfg",
                        help="The configuration file to use.")
    parser.add_argument("--port", type=int, default=8090,
                        help="The TCP port to use")
    args = parser.parse_args()
    config_helper.read_configuration(args.config)
    if args.ranker_name not in scorer_globals.scorers_dict:
        LOG.error("%s is not a valid ranker", args.ranker_name)
        LOG.error("Valid rankers are: %s ", " ".join(
            list(scorer_globals.scorers_dict.keys())))
        sys.exit(1)
    LOG.info("Using ranker %s", args.ranker_name)
    override = json.loads(args.override)
    if override != {}:
        LOG.info('overrides: %s', json.dumps(override))
    ranker_conf = scorer_globals.scorers_dict[args.ranker_name]
    ranker = ranker_conf.instance(override)
    translator = QueryTranslator.init_from_config()
    translator.set_ranker(ranker)

    # using closures prevents us from having to make translator global
    @APP.route('/', methods=['GET'])
    def translate():  # pylint: disable=unused-variable
        """
        REST entry point providing a very simple query interface
        """
        raw_query = flask.request.args.get('q')
        LOG.info("Translating query: %s", raw_query)
        parsed_query, candidates = translator.translate_and_execute_query(
            raw_query)
        LOG.info("Done translating query: %s", raw_query)
        LOG.info("#candidates: %s", len(candidates))
        return flask.jsonify(map_candidates(
            raw_query, parsed_query, candidates))

    @APP.route('/config', methods=['GET'])
    def get_config():
        """
        REST entry point providing information about the current configuration
        """
        result = {
            'ranker_name': ranker_conf.name,
            'override': ranker_conf.override(),
            'config': ranker_conf.config()
            }
        return flask.jsonify(result)

    APP.run(use_reloader=False, host='0.0.0.0', threaded=False,
            port=args.port, debug=False)


if __name__ == "__main__":
    main()
