import argparse
import json


def main(args):
    fname = args.json_file
    with open(fname) as rf, \
            open(f'{fname}.uni', 'w') as wfu, open(f'{fname}.bi', 'w') as wfb, open(f'{fname}.tri', 'w') as wft:
        for line in rf:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = d['source']
            unis = []
            for hypo in d['hypos']:
                uni_score = len([True for c in hypo['text'].replace(' ', '') if c in source])
                hypo['uni_score'] = uni_score
            print(d['id'], file=wfu)
            for hypo in sorted(d['hypos'], key=lambda x: x['uni_score']):
                print(hypo, file=wfu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()
    main(args)
