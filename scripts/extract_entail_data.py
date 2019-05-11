import argparse
import json
import sys

from tqdm import tqdm


def main(args):
    prob_format = args.prob_file.split('.')[-1]
    with open(args.data_file, 'r') as df, open(args.prob_file, 'r') as pf:
        if args.output_file is not None:
            output_file = open(args.output_file, 'w')
        else:
            output_file = sys.stdout
        prob_position = 1 if args.contradiction_first else 0
        for data, prob in tqdm(zip(df, pf)):
            if prob_format == 'tsv':
                entailment_prob = prob.split('¥t')[prob_position]
            elif prob_format == 'json' or prob_format == 'jsonl':
                entailment_prob = json.loads(prob)['label_probs'][prob_position]
            else:
                raise ValueError(f'file format is not supported: {args.data_file}')
            if entailment_prob > 0.5:
                output_file.write(data)
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='path to data file')
    parser.add_argument('prob_file', type=str, help='path to entailment probability file')
    parser.add_argument('-o', '--output-file', type=str, help='path to output file')
    parser.add_argument('-c', '--contradiction-first', action='store_true',
                        help='flag if prob format is contradiction first')
    args = parser.parse_args()
    main(args)
