import argparse
import json

from nltk.util import ngrams


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
            print(d['source'], file=wfu)
            print(d['source'], file=wfb)
            print(d['source'], file=wft)
            print(d['id'], file=wfu)
            print(d['id'], file=wfb)
            print(d['id'], file=wft)
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
                bigram = ngrams(hypo['text'].replace(' ', ''), 2)
                tri_score = len([True for bi in bigram if bi in source_tri])
                hypo['tri_score'] = tri_score

            for hypo in sorted(d['hypos'], key=lambda x: x['uni_score'], reverse=True):
                print(hypo, file=wfu)
            for hypo in sorted(d['hypos'], key=lambda x: x['bi_score']):
                print(hypo, file=wfb)
            for hypo in sorted(d['hypos'], key=lambda x: x['tri_score']):
                print(hypo, file=wft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()
    main(args)
