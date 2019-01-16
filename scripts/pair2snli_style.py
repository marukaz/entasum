import argparse
import json
import sys


def main(args):
    with open(args.premise_file, 'r') as pf, open(args.hypothesis_file, 'r') as hf:
        if args.output_file is not None:
            output_file = open(args.output_file, 'w')
        else:
            output_file = sys.stdout
        for i, (premise, hypo) in enumerate(zip(pf, hf)):
            snli_d = {'id': i, 'sentence1': premise.rstrip(), 'sentence2': hypo.rstrip()}
            print(json.dumps(snli_d, ensure_ascii=False), file=output_file)
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('premise_file', type=str, help='path to sentence1 file')
    parser.add_argument('hypothesis_file', type=str, help='path to sentence2 file')
    parser.add_argument('-o', '--output-file', type=str, help='path to output file')
    args = parser.parse_args()
    main(args)
