import argparse
import json
import sys

from tqdm import tqdm


def main(args):
    with open(args.snli_like_file, 'r') as sf, open(args.prob_file, 'r') as pf:
        if args.output_file is not None:
            output_file = open(args.output_file, 'w')
        else:
            output_file = sys.stdout
        prob_position = 1 if args.contradiction_first else 0
        for data_d, prob_d in tqdm(zip(sf, pf)):
            entailment_prob = json.loads(prob_d)['label_probs'][prob_position]
            if entailment_prob > 0.5:
                output_file.write(data_d)
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('snli_like_file', type=str, help='path to data file')
    parser.add_argument('prob_file', type=str, help='path to entailment probability file')
    parser.add_argument('-o', '--output-file', type=str, help='path to output file')
    parser.add_argument('-c', '--contradiction-first', action='store_true',
                        help='flag if prob format is contradiction first')
    args = parser.parse_args()
    main(args)