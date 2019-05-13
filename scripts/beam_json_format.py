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
            if args.output_format == 'bert':
                snt1 = ''.join(d['source'].split(' '))
                if snt1.startswith('▁'):
                    snt1 = snt1[1:]
            for hypo in d['hypos']:
                if args.output_format == 'esim':
                    snli_d = {'s1_id': d['id'], 'sentence1': d['source'], 'sentence2': hypo['text']}
                    print(json.dumps(snli_d, ensure_ascii=False), file=output_file)
                elif args.output_format == 'bert':
                    snt2 = ''.join(hypo['text'].split(' '))
                    if snt2.startswith('▁'):
                        snt2 = snt2[1:]
                    print(f"{d['id']}\t{snt1}\t{snt2}", file=output_file)
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='path to input file')
    parser.add_argument('--output-format', type=str, choices=['bert', 'esim'], default='bert', help='format of output')
    parser.add_argument('--output-file', type=str, help='path to output file')
    args = parser.parse_args()
    main(args)
