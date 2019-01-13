import argparse
import json


def main(args):
    BEAM_SIZE = 5
    MOD_NUM = BEAM_SIZE - 1
    with open(args.beam_file, 'r') as beamf, open(args.entail_rate_file, 'r') as enf:
        max_prob = 0
        hypos = []
        for i, (beam, rate) in enumerate(zip(beamf, enf)):
            entail_prob = json.loads(rate)['label_probs'][0]
            if entail_prob > max_prob:
                beam_d = json.loads(beam)
                best_hypo = beam_d['sentence2']
                max_prob = entail_prob
            if i % BEAM_SIZE == MOD_NUM:
                hypos.append((int(beam_d['id']), best_hypo))
        for _, hypo in sorted(hypos, key=lambda x:x[0]):
            print(hypo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('beam_file', type=str, help='path to generated hypos file')
    parser.add_argument('entail_rate_file', type=str, help='path to esim style entail rate file')
    args = parser.parse_args()
    main(args)
