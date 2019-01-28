import argparse
import json


def main(args):
    prob_position = 1 if args.contradiction_first else 0
    with open(args.beam_file) as beamf, open(args.entail_rate_file) as enf:
        # get the beam size
        prev_id = json.loads(beamf.readline())['s1_id']
        for i, line in enumerate(beamf, 2):
            current_id = json.loads(line)['s1_id']
            if current_id != prev_id:
                beam_size = i - 1
                mod_num = beam_size - 1
                beamf.seek(0)
                break
            prev_id = current_id

        max_prob = -1
        hypos = []
        for i, (beam, rate) in enumerate(zip(beamf, enf)):
            entail_prob = json.loads(rate)['label_probs'][prob_position]
            if entail_prob > max_prob:
                beam_d = json.loads(beam)
                rerank_best = beam_d['sentence2']
                if max_prob == -1:
                    conventional_best = rerank_best
                max_prob = entail_prob
            if i % beam_size == mod_num:
                hypos.append((int(beam_d['s1_id']), rerank_best, conventional_best))
                max_prob = -1
        with open(f'{args.beam_file}.reranked', 'w') as rankf, open(f'{args.beam_file}.conventional', 'w') as convf:
            for _, rerank, conventional in sorted(hypos, key=lambda x: x[0]):
                print(''.join(rerank.split(' ')[1:]), file=rankf)
                print(''.join(conventional.split(' ')[1:]), file=convf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('beam_file', type=str, help='path to generated hypos file')
    parser.add_argument('entail_rate_file', type=str, help='path to esim style entail rate file')
    parser.add_argument('-c', '--contradiction-first', action='store_true',
                        help='flag if prob format is contradiction first')
    args = parser.parse_args()
    main(args)
