import argparse
import json


def main(args):
    with open(args.load, 'r') as f:
        unique_hypos = []
        texts = []
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for hypo in d['hypos']:
                text = hypo['text']
                if text not in texts:
                    unique_hypos.append(hypo)
                    texts.append(text)
            d['hypos'] = unique_hypos
            print(json.dumps(d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load')
    args = parser.parse_args()
    main(args)
