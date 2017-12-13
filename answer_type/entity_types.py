#!/usr/bin/env python3
import argparse
import logging
import sys
from collections import Counter, defaultdict
from operator import itemgetter

remove_types = {
        'award.award_winner',
        'award.award_nominee',
        'award.ranked_item',
        'film.film_subject',
        'book.book_subject',
        'business.issuer',
        'award.award_nominated_work',
        'award.award_winning_work',
        'radio.radio_subject',
        'book.published_work',
        'book.published_work'
        'music.release_trac',
        'music.recording'
        }
remove_prefixes = (
        'common.',
        'base.',
        'type.',
        'user.',
        )

def extract_mid(eurl):
    return eurl[eurl.rfind('/')+1:eurl.rfind('>')]

def write_tsv_line(outfile, *args):
    """
    Write a tab separated values file line with
    the columns given by args
    """
    outfile.write('\t'.join(args)+'\n')

def gen_type_counts(in_filename, out_filename):
    """
    Given a file of the form:
    <prefix/mid>\\t<prefix/typename>\\n
    generates a type count mapping file of the form
    typename\\tCOUNT\\n
    """
    counter = Counter()
    with open(in_filename, 'rt', encoding='utf-8') as infile:
        for line in infile:
            eurl, etype = line.split('\t')
            etype = extract_mid(etype)
            counter[etype]+= 1

    with open(out_filename, 'wt', encoding='utf-8') as outfile:
        for etype, count in counter.most_common():
            write_tsv_line(outfile, etype, str(count))

def clean_type_list(type_list, type_counter):
    global remove_types
    global remove_prefixes

    res_list = []
    for t in type_list:
        if t not in remove_types and not t.startswith(remove_prefixes):
            res_list.append((t, type_counter[t]))
        else:
            res_list.append((t, 0))

    return sorted(res_list, 
            key=itemgetter(1),
            reverse=True)

def gen_entity_types_cleaned(in_filename, out_filename):
    """
    Given a file of the form:
    <prefix/mid>\t<prefix/typename>\n
    (sorted by the first column)
    generates a mapping from entity to types of the form
    mid\ttype0\ttype1\t...\n
    where the types are sorted by popularity with non specific types removed
    """
    entity_types_map = defaultdict(list)
    counter = Counter()
    with open(in_filename, 'rt', encoding='utf-8') as infile:
        for line in infile:
            eurl, etype = line.split('\t')
            emid = extract_mid(eurl)
            etype = extract_mid(etype)
            counter[etype] += 1
            entity_types_map[emid].append(etype)


    with open(out_filename, 'wt', encoding='utf-8') as outfile:
        for emid, etypes in entity_types_map.items():
            type_list = [t for t, _ in clean_type_list(etypes, counter)]
            write_tsv_line(outfile, emid, 
                    ' '.join(type_list))



def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['type_counts', 'entity_types'])
    parser.add_argument('file')
    parser.add_argument('savefile')
    args = parser.parse_args()

    if args.mode == 'type_counts':
        gen_type_counts(args.file, args.savefile)
    elif args.mode == 'entity_types':
        gen_entity_types_cleaned(args.file, args.savefile)


if __name__ == "__main__":
    main()
