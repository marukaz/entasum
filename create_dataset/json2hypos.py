import argparse
import json


def main(args):
    with open(args.load, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for hypo in d['hypos']:
                print(hypo['text'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load')
    args = parser.parse_args()
    main(args)
