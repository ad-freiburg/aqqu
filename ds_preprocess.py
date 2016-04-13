"""Pre-process distant supervision data to generate a file with
text <tab> relation
lines.
"""
import logging
import util
import re
import random
import sys
import gzip


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

brackets = re.compile(r"(\([^\(]+\))")

number = re.compile(r" (\d+) ")

ignore_relations = {"film.film_regional_release_date.film_release_region+film.film_regional_release_date.film_release_region",
                    "film.film_regional_release_date.film_release_region+film.film_regional_release_date.film_release_region+film.film_regional_release_date.film_release_region",
                    "military.military_combatant_group.combatants+military.military_combatant_group.combatants",
                    "military.military_combatant_group.combatants+military.military_combatant_group.conflict",
                    }
ignore_relation_prefixes = {"location.", "base.", "user.",
                            "time.", "influence.influence_node",
                            "measurement_unit.rect_size",
                            "common.phone_number.",
                            "basketball"}


def replace_entities_in_sentence(sentence, entities):
    def replace(match):
        if (match.group(1)) in entities:
            return "<entity>"
        else:
            return match.group(2)
    s = re.sub(util.entity_sentence_re, replace, sentence)
    return s


def read_and_write_ds_examples(sentences_file, output_file, max_span=10):
    """Read the DS examples from file. Return a list of tuples
    ([relations], text).

    :param sentences_file:
    :return:
    """
    n_lines = 0
    label_set = set()
    num_examples = 0
    logger.info("Writing to %s" % output_file)
    with open(output_file, "w") as out_f:
        with gzip.open(sentences_file, "rt") as f:
            for line in f:
                cols = line.strip().split('\t')
                sentence = cols[0]
                n_lines += 1
                if n_lines % 100000 == 0:
                    logger.info("Processed %d lines." % n_lines)
                    continue
                if len(cols) < 4:
                    continue
                relations = cols[3]
                ## Get subj, relation , obj
                rel_parts = relations.split('&&&')
                for r in rel_parts:
                    r = r.strip()
                    m = util.binary_rel_re.match(r)
                    mm = util.binary_mediated_rel_re.match(r)
                    mmm = util.ternary_mediated_rel_re.match(r)
                    ignore = False
                    entities = []
                    if m:
                        subj = m.group(1)
                        rel = m.group(3)
                        obj = m.group(4)
                        entities = [subj, obj]
                    elif mm:
                        subj = mm.group(1)
                        obj = mm.group(4)
                        rels = sorted([mm.group(3), mm.group(6)])
                        rel = "+".join(rels)
                        entities = [subj, obj]
                    elif mmm:
                        subj = mmm.group(1)
                        obj = mmm.group(4)
                        obj2 = mmm.group(7)
                        entities = [subj, obj, obj2]
                        rels = sorted([mmm.group(3), mmm.group(6), mmm.group(9)])
                        rel = "+".join(rels)
                    else:
                        logger.warn("Couldn't parse relation format %s." % r)
                        continue
                    example_sentence = replace_entities_in_sentence(sentence,
                                                                    entities)
                    # Check if we ignore this relation.
                    # Relations with suffix "REVERSE" are from entity to mediator,
                    # but have no reverese relation from mediator to entity. We
                    # ignore those for now.
                    if rel in ignore_relations or "REVERSE" in rel:
                        continue
                    else:
                        for prefix in ignore_relation_prefixes:
                            if rel.startswith(prefix):
                                ignore = True
                                break
                        if ignore:
                            continue
                    # Remove brackets.
                    example_sentence = re.sub(brackets, "", example_sentence)
                    example_sentence = re.sub(number, " <num> ", example_sentence)
                    window_size = 3
                    if example_sentence.count("<entity>") > 1:
                        tokens = example_sentence.split(' ')
                        last_index = len(tokens) - 1 - tokens[::-1].index("<entity>")
                        start = max(0, tokens.index('<entity>') - window_size)
                        end = last_index + window_size + 1
                        span = tokens[start:end]
                        if len(span) <= max_span:
                            span = " ".join(span)
                            out_f.write("%s\t%s\n" % (span, rel))
                            label_set.add(rel)
                            num_examples += 1
                        else:
                            pass
        logger.info("Read %d examples for %d classes." % (num_examples,
                                                          len(label_set)))


def main():
    if len(sys.argv) != 3:
        print("Usage: <script> sentences_file.txt.gz outputfile")
        exit(1)
    sentences_file, output_file = sys.argv[1], sys.argv[2]
    read_and_write_ds_examples(sentences_file,
                               output_file)

if __name__ == '__main__':
    main()
