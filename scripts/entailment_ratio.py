import argparse
import json

from tqdm import tqdm


def main(args):
    all_num = 0
    entail_num = 0
    with open(args.prob_file) as pf:
        prob_position = 1 if args.contradiction_first else 0
        for prob in tqdm(pf):
            entailment_prob = float(prob.split('\t')[prob_position])
            if entailment_prob > 0.5:
                entail_num += 1
            all_num += 1
    print(entail_num/all_num)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prob_file', type=str, help='path to entailment probability file')
    parser.add_argument('-c', '--contradiction-first', action='store_true',
                        help='flag if prob format is contradiction first')
    args = parser.parse_args()
    main(args)
