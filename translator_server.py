"""
Provides a REST based translation interface.

Copyright 2017, University of Freiburg.

Niklas Schnelle <schnelle@cs.uni-freiburg.de>

"""
import logging
from typing import List, Union
import config_helper
import scorer_globals
import flask
import spacy
from query_translator.translator import QueryTranslator, Query
import entity_linker.entity_linker as entity_linker
import freebase

logging.basicConfig(format="%(asctime)s : %(levelname)s "
                           ": %(module)s : %(message)s",
                    level=logging.DEBUG)

LOG = logging.getLogger(__name__)


APP = flask.Flask(__name__)


def map_results_list(results: List[List[Union[str, int]]]) -> List[dict]:
    """
    Maps the result rows of the SPARQL backend into JSON compatible lists
    """
    query_results = []
    for result in results:
        if len(result) > 1:
            query_results.append({'value': result[1], 'mid': result[0]})
        else:
            query_results.append({'value': result[0]})
    return query_results


def map_query_doc(doc: spacy.tokens.Doc) -> List[dict]:
    """
    Maps a spaCy Doc into a JSON compatible list
    """
    return [{'orth': tok.orth_, 'tag': tok.tag_,
             'offset': tok.idx} for tok in doc]


def map_entity(entity: entity_linker.Entity) -> dict:
    """
    Maps an entity_linker.Entity to a JSON compatible dict
    """
    return {'name': entity.name,
            'mid': entity.sparql_name(),
            'sparql': entity.prefixed_sparql_name(freebase.FREEBASE_NS_PREFIX)}


def map_identified_entity(entity: entity_linker.IdentifiedEntity) -> dict:
    """
    Maps an IdentifiedEntity into a JSON compatible dict
    """
    mapped_entity = {
        'name': entity.name,
        'token_positions': [tok.i for tok in entity.tokens],
        'surface_score': entity.surface_score,
        'score': entity.score,
        'entity': map_entity(entity.entity),
        'perfect_match': entity.perfect_match,
        'text_match': entity.text_match,
        'types': entity.types
    }
    return mapped_entity


def map_query(query: Query) -> dict:
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


def map_translations(raw_query, results) -> dict:
    """
    Turns the final translations into a list of candidate maps suitable
    for json encoding
    """
    candidates = []
    for result in results:
        candidate = result.query_candidate
        query_results = map_results_list(result.query_result_rows)

        candidates.append({
            'query': map_query(candidate.query),
            'sparql': candidate.to_sparql_query(),
            'matches_answer_type': candidate.matches_answer_type,
            'pattern': candidate.pattern,
            'graph': candidate.graph_as_simple_string(indent=1),
            'results': query_results,
            })

    return {'raw_query': raw_query, 'candidates': candidates}


def main() -> None:
    """
    Entry point into the program
    """
    import argparse
    parser = argparse.ArgumentParser(description="REST api based translation.")
    parser.add_argument("ranker_name",
                        default="WQ_Ranker",
                        help="The ranker to use.")
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
    LOG.info("Using ranker %s", args.ranker_name)
    ranker = scorer_globals.scorers_dict[args.ranker_name]
    translator = QueryTranslator.init_from_config()
    translator.set_scorer(ranker)

    # using a closure prevents us from having to make translator global
    @APP.route('/', methods=['GET'])
    def translate():  # pylint: disable=unused-variable
        """
        REST entry point providing a very simple query interface
        """
        query = flask.request.args.get('q')
        LOG.info("Translating query: %s", query)
        translations = translator.translate_and_execute_query(query)
        LOG.info("Done translating query: %s", query)
        LOG.info("#candidates: %s", len(translations))
        return flask.jsonify(map_translations(query, translations))

    APP.run(use_reloader=False, host='0.0.0.0', threaded=False,
            port=args.port, debug=False)


if __name__ == "__main__":
    main()
