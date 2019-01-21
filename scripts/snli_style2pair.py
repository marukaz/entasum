import argparse
import json


def main(args):
    out_prefix = args.output_prefix or 'out'
    src_path = out_prefix + '.src'
    tgt_path = out_prefix + '.tgt'
    with open(args.snli_like_file, 'r') as rf, open(src_path, 'w') as sf, open(tgt_path, 'w') as tf:
        for line in rf:
            d = json.loads(line)
            if ((args.label is None or d['gold_label'] == args.label)
                    and (args.type is None or d['headline_type'] == args.type)):
                print(d['sentence1'], file=sf)
                print(d['sentence2'], file=tf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('snli_like_file', type=str, help='path to snli format jsonl file')
    parser.add_argument('-o', '--output-prefix', type=str, help="output files name's prefix")
    parser.add_argument('-l', '--label', choices=['entailment', 'contradiction'],
                        help="output only the specified label")
    parser.add_argument('-t', '--type', choices=['reference', 'best', 'none'],
                        help="output only the specified type")
    args = parser.parse_args()
    main(args)
