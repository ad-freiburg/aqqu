#!/usr/bin/env python3
"""
Small script to convert Natalie Prange's generated question format which uses
Freebase Easy names and Types to a minimal Freebase linked format.

This source format uses entity annotations of the form
[<easy_name>|<easy_type>|<tokens>] where tokens are separated by ' '

In our output format each entity is represented as [<mid>|<tokens>] where
tokens are separated by '_' to allow splitting by ' '. If the <easy_type>
was 'Year' in the input an artificial mid '<VALUE>' is used instead.
"""
import argparse
import logging
from collections import defaultdict


def extract_mid(eurl):
    """
    Extracts the mid from a '<http://freebase../ns/mid>' URL
    """
    return eurl[eurl.rfind('/')+1:eurl.rfind('>')]


def load_freebase_easy_links(links_path):
    """
    Loads the Freebase Easy Links file for mapping Freebase Easy
    names to Freebase mids
    """
    logging.info('Loading Freebase Easy Links file')
    name_to_mid = defaultdict(lambda: 'UNK')
    with open(links_path, 'rt', encoding='utf-8') as links_file:
        for line in links_file:
            name, _, url, _ = line.split('\t')
            name_to_mid[name] = extract_mid(url)
    logging.info('Freebase Easy Links file loaded')
    return name_to_mid

def load_name_to_mid(name_to_mid_path):
    name_to_mid = defaultdict(lambda: 'UNK')
    with open(name_to_mid_path, 'rt', encoding='utf-8') as name_to_mid_file:
        for line in name_to_mid_file:
            name, mid = line.split('\t')
            name_to_mid[name] = mid.strip()
    return name_to_mid


def freebasesize_mention(mention, name_to_mid):
    """
    Converts a mention in Natalie's input format to a (mid, token)
    tuple where for <easy_type> == 'Year' the mid will be '<VALUE>'
    """
    mention = mention[1:-1]
    splits = mention.split('|')
    name = splits[0].replace('_', ' ')
    easy_type = splits[1]
    if easy_type == 'Year':
        mid = '<VALUE>'
    else:
        mid = name_to_mid[name]
    tokens = splits[2].split(' ')
    return (mid, tokens)


def mention_to_str(mention):
    """
    Formats a (mid, tokens) tuple as a str
    """
    return '[{}|{}]'.format(mention[0], '_'.join(mention[1]))


def gq_freebaseize(in_path, name_to_mid, out_path):
    """
    Converts a file in Natalie's generated question format to
    our Freebase simple question format
    """
    with open(in_path, 'rt', encoding='utf-8', errors='replace') as gq_file,\
            open(out_path, 'wt', encoding='utf-8') as out_file:
        for line in gq_file:
            question, answer = line.split('\t')
            outtoks = []
            in_mention = False
            mention_str = ''
            for tok in question.split(' '):
                if in_mention:
                    mention_str += ' '+tok
                    if tok.endswith(']'):
                        in_mention = False
                        outtoks.append(mention_to_str(freebasesize_mention(
                            mention_str, name_to_mid)))
                else:
                    if tok.startswith('['):
                        mention_str = tok
                        if tok.endswith(']'):
                            outtoks.append(mention_to_str(freebasesize_mention(
                                mention_str, name_to_mid)))
                        else:
                            in_mention = True
                    else:
                        outtoks.append(tok)

            answer_mention = freebasesize_mention(answer, name_to_mid)
            out_file.write(' '.join(outtoks) +
                           '\t'+mention_to_str(answer_mention)+'\n')


def main():
    """
    Provides a very simple commandline arguments interface
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--name-to-mid', default='data/name-to-mid.txt')
    parser.add_argument('savefile')
    args = parser.parse_args()

    name_to_mid = load_name_to_mid(args.name_to_mid)

    gq_freebaseize(args.file, name_to_mid, args.savefile)


if __name__ == "__main__":
    main()
