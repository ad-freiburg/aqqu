"""
Provides a REST based translation interface.

Copyright 2015, University of Freiburg.

Niklas Schnelle <schnelle@cs.uni-freiburg.de>

"""
import logging
import config_helper
import scorer_globals
import flask
from query_translator.translator import QueryTranslator

logging.basicConfig(format="%(asctime)s : %(levelname)s "
                           ": %(module)s : %(message)s",
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


app = flask.Flask(__name__)

def map_results_list(results):
    """
    Turns the result rows of the SPARQL backend into JSON compatible lists
    """
    query_results = []
    for r in results:
        if len(r) > 1:
            query_results.append({'value': r[1], 'mid': r[0]})
        else:
            query_results.append({'value' : r[0]})
    return query_results


class Query:
    """
    A query that is to be translated.
    """

    def __init__(self, text):
        self.query_text = text.lower()
        self.target_type = None
        self.query_tokens = None
        self.query_content_tokens = None
        self.identified_entities = None
        self.relation_oracle = None
        self.is_count_query = False

def map_token(tok):
    """
    Turns a spaCy Token into a JSON compatible map
    """
    return {'orth': tok.orth_, 'tag': tok.tag_, 'idx': tok.idx}

def map_query(query):
    """
    Turns a Query into a JSON compatible map
    """
    tokens = [map_token(tok) for tok in query.query_tokens]
    content_tokens = [map_token(tok) for tok in query.query_content_tokens]

    identified_entities = []
    query_map = {
        'target_type': query.target_type.as_string(),
        'tokens': tokens,
        'content_tokens': content_tokens,
        'identified_entities': identified_entities,
        'is_count': query.is_count_query
    }
    return query_map

def map_translations(raw_query, results):
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


def main():
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
        logger.error("%s is not a valid ranker", args.ranker_name)
        logger.error("Valid rankers are: %s ", " ".join(list(scorer_globals.scorers_dict.keys())))
    logger.info("Using ranker %s", args.ranker_name)
    ranker = scorer_globals.scorers_dict[args.ranker_name]
    translator = QueryTranslator.init_from_config()
    translator.set_scorer(ranker)

    # using a closure prevents us from having to make translator global
    @app.route('/', methods=['GET'])
    def translate():
        """
        REST entry point providing a very simple query interface
        """
        query = flask.request.args.get('q')
        logger.info("Translating query: %s", query)
        translations = translator.translate_and_execute_query(query)
        logger.info("Done translating query: %s", query)
        logger.info("#candidates: %s", len(translations))
        return flask.jsonify(map_translations(query, translations))


    app.run(use_reloader=False, host='0.0.0.0', threaded=False,
            port=args.port, debug=False)

if __name__ == "__main__":
    main()
