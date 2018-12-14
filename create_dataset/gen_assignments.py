import argparse
import json
import numpy as np


def main(args):
    assignments = np.load('/home/6/18M31289/entasum/create_dataset/generate_candidates/data/assignments-pretrained.npy')
    with open(args.json_file) as f:
        for a, line in zip(assignments, f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for ix in a:
                print(d['hypos'][ix])
            print('------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()
    main(args)
