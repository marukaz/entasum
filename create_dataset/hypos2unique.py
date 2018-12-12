import argparse
import json


def main(args):
    with open(args.load, 'r') as f:
        for line in f:
            unique_hypos = []
            texts = []
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
            print(json.dumps(d, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load')
    args = parser.parse_args()
    main(args)
