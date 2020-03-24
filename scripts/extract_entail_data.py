import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def main(args):
    prob_format = args.prob_file.split('.')[-1]
    src_name = args.data_src.split('/')[-1]
    tgt_name = args.data_tgt.split('/')[-1]
    with open(args.data_src) as dsf, open(args.data_tgt) as dtf, open(args.prob_file) as pf:
        output_src = open(Path(args.output_dir)/src_name, 'w')
        output_tgt = open(Path(args.output_dir)/tgt_name, 'w')
        if args.not_entail_dir is not None:
            not_entail_src = open(Path(args.not_entail_dir)/src_name, 'w')
            not_entail_tgt = open(Path(args.not_entail_dir)/tgt_name, 'w')
        prob_position = 1 if args.contradiction_first else 0
        for src, tgt, prob in tqdm(zip(dsf, dtf, pf)):
            if prob_format == 'tsv':
                entailment_prob = float(prob.split('\t')[prob_position])
            elif prob_format == 'json' or prob_format == 'jsonl':
                entailment_prob = json.loads(prob)['label_probs'][prob_position]
            else:
                raise ValueError(f'file format is not supported: {args.prob_file}')
            if entailment_prob > 0.5:
                output_src.write(src)
                output_tgt.write(tgt)
            else:
                if args.not_entail_dir is not None:
                    not_entail_src.write(src)
                    not_entail_tgt.write(tgt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_src', type=str, help='path to data file')
    parser.add_argument('data_tgt', type=str, help='path to data file')
    parser.add_argument('prob_file', type=str, help='path to entailment probability file')
    parser.add_argument('-o', '--output-dir', type=str, help='path to output dir')
    parser.add_argument('--not-entail-dir', type=str, default=None,
                        help='path to output not entail data')
    parser.add_argument('-c', '--contradiction-first', action='store_true',
                        help='flag if prob format is contradiction first')
    args = parser.parse_args()
    main(args)
