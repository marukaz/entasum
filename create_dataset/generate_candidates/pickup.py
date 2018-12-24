import argparse
import random


def main(args):
    random.seed(123)
    with open(args.load, 'r') as f:
        unit_list = [unit[:-1] for unit in zip(*[iter(f)] * args.lines_per_unit)]
    for unit in random.sample(unit_list, args.sample_num):
        print(*unit)
        print('************************************************************************************************')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('load')
    parser.add_argument('--lines-per-unit', type=int, default=7)
    parser.add_argument('--sample-num', type=int, default=50)
    args = parser.parse_args()
    main(args)
