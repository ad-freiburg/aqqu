"""Pre-process the 30m questions file to a format with
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


def replace_entities_in_sentence(sentence, entities):
    def replace(match):
        if (match.group(1)) in entities:
            return "<entity>"
        else:
            return match.group(2)
    s = re.sub(util.entity_sentence_re, replace, sentence)
    return s


def get_subj_mid(subj):
    #prefix = "<http://rdf.freebase.com/ns/"
    return subj[28:-1]

def get_relation_name(rel):
    #prefix = www.freebase.com/
    if rel.startswith("<"):
        return rel[28:-1].replace('/', '.')
    else:
        return rel[17:].replace('/', '.')

def transform_30m(input_file, output_file, name_mid):
    with open(output_file, "w") as out:
        with open(input_file) as f:
            for line in f:
                cols = line.strip().split('\t')
                subj_mid = get_subj_mid(cols[0])
                question = cols[3]
                if subj_mid in name_mid:
                    name = name_mid[subj_mid]
                    if name in question:
                        # Replace the entity in the question with <entity>
                        question = question.replace(name, subj_mid.replace('.',
                                                                           '_'))
                        relation = get_relation_name(cols[1])
                        out.write("%s\t%s\n" % (question, relation))
                    else:
                        pass
                else:
                    pass
                    #print(subj_mid)



def read_entity_list(entity_list_file):
    mid_name = {}
    print("Reading mid names from %s" % entity_list_file)
    suffix = re.compile(r" \(.*\).*$")
    with open(entity_list_file) as f:
        for line in f:
            cols = line.strip().split('\t')
            mid = cols[0]
            name = cols[1].lower()
            name = re.sub(suffix, "", name)
            name = name.replace(', ', ' , ')
            mid_name[mid] = name
            if len(mid_name) % 1000000 == 0:
                print("Read %d mid names" % len(mid_name))
    print("Read %d mid names" % len(mid_name))
    return mid_name


def main():
    if len(sys.argv) != 4:
        print("Usage: <script> entity_list 30m_file output_file")
        exit(1)
    entity_list, input_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    name_mid = read_entity_list(entity_list)
    transform_30m(input_file, output_file, name_mid)

if __name__ == '__main__':
    main()
