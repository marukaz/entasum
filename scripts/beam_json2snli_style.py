import argparse
import json
import sys


def main(args):
    with open(args.input_file, 'r') as f:
        if args.output_file is not None:
            output_file = open(args.output_file, 'w')
        else:
            output_file = sys.stdout
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(e, line)
            for hypo in d['hypos']:
                snli_d = {'s1_id': d['id'], 'sentence1': d['source'], 'sentence2': hypo['text']}
                print(json.dumps(snli_d, ensure_ascii=False), file=output_file)
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='path to input file')
    parser.add_argument('--output-file', type=str, help='path to output file')
    args = parser.parse_args()
    main(args)
