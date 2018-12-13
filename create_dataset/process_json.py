import argparse
import json
import random

import numpy as np


def main(args):
    r"""
    正解文がhyposの先頭にあるという前提のもと、正解文が一番長くなければ、
    hyposをユニークにし、threashold以上の候補があればその数まで標準出力にjson形式で書き出す
    :param args:
    :return:
    """
    with open(args.load, 'r') as rf, open(f'{args.load}.unique_th{args.threshold}', 'w') as wf:
        for line in rf:
            unique_hypos = []
            texts = []
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            longest = np.argmax([len(hypo['text']) for hypo in d['hypos']])
            if longest != 0:
                for hypo in d['hypos']:
                    text = hypo['text']
                    if text not in texts:
                        unique_hypos.append(hypo)
                        texts.append(text)
                if len(unique_hypos) >= args.threshold:
                    d['hypos'] = random.sample(unique_hypos[1:], args.threshold-1).insert(unique_hypos[0], 0)
                    print(json.dumps(d, ensure_ascii=False), file=wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load')
    parser.add_argument('-t', '--threshold', type=int, default=32)
    args = parser.parse_args()
    main(args)
