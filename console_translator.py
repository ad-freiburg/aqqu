"""
Provides a console based translation interface.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging
import config_helper
import scorer_globals
import sys
from query_translator.translator import QueryTranslator

logging.basicConfig(format="%(asctime)s : %(levelname)s "
                           ": %(module)s : %(message)s",
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Console based translation.")
    parser.add_argument("ranker_name",
                        default="WQ_Ranker",
                        help="The ranker to use.")
    parser.add_argument("--config",
                        default="config.cfg",
                        help="The configuration file to use.")
    args = parser.parse_args()
    config_helper.read_configuration(args.config)
    if args.ranker_name not in scorer_globals.scorers_dict:
        logger.error("%s is not a valid ranker" % args.ranker_name)
        logger.error("Valid rankers are: %s " % (" ".join(list(scorer_globals.scorers_dict.keys()))))
    logger.info("Using ranker %s" % args.ranker_name)
    ranker = scorer_globals.scorers_dict[args.ranker_name].instance()
    translator = QueryTranslator.init_from_config()
    translator.set_ranker(ranker)
    while True:
        sys.stdout.write("enter question> ")
        sys.stdout.flush()
        query = sys.stdin.readline().strip()
        logger.info("Translating query: %s" % query)
        _, candidates = translator.translate_and_execute_query(query)
        logger.info("Done translating query: %s" % query)
        logger.info("#candidates: %s" % len(results))
        if len(candidates) > 0:
            best_candidate = results[0]
            sparql_query = best_candidate.to_sparql_query()
            result_rows = best_candidate.query_result
            result = []
            # Usually we get a name + mid.
            for r in result_rows:
                if len(r) > 1:
                    result.append("%s (%s)" % (r[1], r[0]))
                else:
                    result.append("%s" % r[0])
            logger.info("SPARQL query: %s" % sparql_query)
            logger.info("Result: %s " % " ".join(result))


if __name__ == "__main__":
    main()
