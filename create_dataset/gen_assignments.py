import argparse
import json
import numpy as np


def main():
    assignments = np.load('/home/6/18M31289/entasum/create_dataset/generate_candidates/data/assignments-pretrained.npy')
    with open('/home/6/18M31289/entasum/create_dataset/generate_candidates/data/data.json') as f:
        for a, line in zip(assignments, f):
            d = json.loads(line)
            for ix in a:
                print(d['hypos'][ix])
            print('--------------------------------------------------')


if __name__ == "__main__":
    main()
