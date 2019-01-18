import argparse
import json
import sys


def main(args):
    with open(args.snli_like_file, 'r') as sf, open(args.prob_file, 'r') as pf:
        if args.output_file is not None:
            output_file = open(args.output_file, 'w')
        else:
            output_file = sys.stdout
        if args.contradiction_first:
            prob_position = 1
        else:
            prob_position = 0
        for data_d, prob_d in zip(sf, pf):
            entailment_prob = json.loads(prob_d)['label_probs'][prob_position]
            if entailment_prob > 0.5:
                print(data_d, file=output_file)
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('snli_like_file', type=str, help='path to data file')
    parser.add_argument('prob_file', type=str, help='path to entailment probability file')
    parser.add_argument('-o', '--output-file', type=str, help='path to output file')
    parser.add_argument('-c', '--contradiction-first', action='store_false',
                        help='flag if prob format is contradiction first')
    args = parser.parse_args()
    main(args)
