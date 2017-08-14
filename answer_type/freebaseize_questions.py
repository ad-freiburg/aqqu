#!/usr/bin/env python3
import argparse
import logging
import re
import sys
from collections import Counter, defaultdict
from operator import itemgetter

mentionregex = re.compile(r'\[[^\|\]]+?\|[^\]]+?\]')

class EntityMention:
    def __init__(self, name=None, easy_type='UNK', 
            mid='UNK', types=['UNK'], position=0):
        self.name = name
        self.mid = mid
        self.types = types
        self.easy_type = easy_type
        self.tokens = name.split('_')
        self.span = (position, position+len(self.tokens))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "[{}|{}|{}|{}]".format(self.name,
                self.easy_type,
                self.mid, ','.join(self.types)) 

    @staticmethod
    def fromString(text, position=0):
        if text[0] != '[' and text[-1] != ']':
            return EntityMention(name=text)
        else:
            splits = text[1:-1].split('|')
            name = splits[0]
            easy_type = splits[1]
            mid = splits[2]
            types = splits[3].split(',')
            return EntityMention(name, easy_type, mid, types, position)

def load_name_to_mid(name_to_mid_path):
    name_to_mid = defaultdict(lambda: 'UNK')
    with open(name_to_mid_path, 'rt', encoding='utf-8') as name_to_mid_file:
        for line in name_to_mid_file:
            name, mid = line.split('\t')
            name_to_mid[name] = mid.strip()
    return name_to_mid


def load_entity_types(entity_types_path, max_len=None):
    entity_types_map = defaultdict(lambda: ['UNK'])
    with open(entity_types_path, 'rt', encoding='utf-8') as entity_types_file:
        for line in entity_types_file:
            mid, types = line.split('\t')
            types = types.strip()
            # list[:None] is the same as list[:] see
            # https://stackoverflow.com/q/30622809
            entity_types_map[mid] = types.split(' ')[:max_len]
    return entity_types_map

def freebasesize_mention(mention, name_to_mid, entity_types_map, position=0):
    mention = mention[1:-1]
    splits = mention.split('|')
    name = splits[0]
    easy_type = splits[1]
    mid = name_to_mid[name]
    return EntityMention(name, easy_type, mid,
            entity_types_map[mid], position)

def freebasesize_answer(answer, name_to_mid, entity_types_map):
    answer = answer.strip()
    if answer[0] == '[' and answer[-1] == ']':
        return freebasesize_mention(answer, name_to_mid, entity_types_map)
    else:
        return EntityMention(name=answer, easy_type='year')

def dumb_tokenize(text):
    toks_split = text.split(' ')
    build_tok = None
    for tok in toks_split:
        if not build_tok:
            if tok[0] != '[' or tok[-1] == ']':
                yield tok
            else:
                build_tok = tok
        else:
            if tok[-1] == ']':
                yield build_tok+tok
                build_tok = None
            else:
                build_tok += ' '+tok

def gq_freebaseize(gq_file_path, name_to_mid, entity_types_map, outfile_path):
    with open(gq_file_path, 'rt', encoding='utf-8', errors='replace') as gq_file,\
        open(outfile_path, 'wt', encoding='utf-8') as out_file:
        for line in gq_file:
            question, answer = line.split('\t')
            answer_entity = freebasesize_answer(answer, name_to_mid, entity_types_map)
            outtoks = []

            for tok in dumb_tokenize(question):
                match = mentionregex.match(tok)
                if match:
                    em = freebasesize_mention(match.string, name_to_mid,
                            entity_types_map)
                    outtoks.append(str(em))
                else:
                    outtoks.append(tok)
            out_file.write(' '.join(outtoks))
            out_file.write('\t'+str(answer_entity)+'\n')




def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--name_to_mid', default='data/name_to_mid.txt')
    parser.add_argument('--entity_types', default='data/entity_types_clean.tsv')
    parser.add_argument('savefile')
    args = parser.parse_args()

    name_to_mid = load_name_to_mid(args.name_to_mid)
    entity_types_map = load_entity_types(args.entity_types, max_len=1)

    gq_freebaseize(args.file, name_to_mid, entity_types_map, args.savefile)


if __name__ == "__main__":
    main()
