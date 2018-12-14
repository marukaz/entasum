import argparse
import json

from nltk.util import ngrams


def print2file(f, d, key_score, reverse=False):
    print(d['source'], file=f)
    print(d['id'], file=f)
    print('text\tuni\tbi\ttri')
    for hypo in sorted(d['hypos'], key=lambda x: x[key_score], reverse=reverse):
        print(hypo['text'], hypo['uni_score'], hypo['bi_score'], hypo['tri_score'], sep='\t', file=f)


def main(args):
    fname = args.json_file
    with open(fname) as rf, \
            open(f'{fname}.uni', 'w') as wfu, open(f'{fname}.bi', 'w') as wfb, open(f'{fname}.tri', 'w') as wft:
        for line in rf:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_concat = d['source'].replace(' ', '')
            for hypo in d['hypos']:
                uni_score = len([True for c in hypo['text'].replace(' ', '') if c in source_concat])
                hypo['uni_score'] = uni_score

            source_bi = set(ngrams(source_concat, 2))
            for hypo in d['hypos']:
                bigram = ngrams(hypo['text'].replace(' ', ''), 2)
                bi_score = len([True for bi in bigram if bi in source_bi])
                hypo['bi_score'] = bi_score

            source_tri = set(ngrams(source_concat, 3))
            for hypo in d['hypos']:
                trigram = ngrams(hypo['text'].replace(' ', ''), 3)
                tri_score = len([True for tri in trigram if tri in source_tri])
                hypo['tri_score'] = tri_score

            print2file(wfu, d, 'uni_score', reverse=True)
            print2file(wfb, d, 'bi_score')
            print2file(wft, d, 'tri_score')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()
    main(args)
