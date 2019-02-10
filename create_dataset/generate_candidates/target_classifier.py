import argparse
import json
from operator import itemgetter
from pathlib import Path
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from tqdm import tqdm

from create_dataset.generate_candidates.util import feature_extractor


def main(args):
    # TODO: treatment of data is not elegant
    X, batch_size, gen_scores, corpus, sources = feature_extractor(args)
    y = [0]*len(gen_scores)
    for i in range(len(y)):
        if i % batch_size == 0:
            y[i] = 1

    if args.mode == 'train':
        clf = LogisticRegression()
        print('start to learn')
        clf.fit(X, y)
        print(clf.score(X, y))
        joblib.dump(clf, args.clf_name)
    elif args.mode == 'eval':
        clf = joblib.load(args.clf_name)
        probas = clf.predict_proba(X)
        corpus_batch_itr = zip(*[iter(corpus)] * batch_size)
        probas_batch_itr = zip(*[iter(probas)] * batch_size)
        for snt_b, proba_b, src in zip(corpus_batch_itr, probas_batch_itr, sources):
            print(f'source: {src}')
            batch_id = range(batch_size)
            batch = np.column_stack((batch_id, snt_b, proba_b))
            for id_, snt, _, proba in sorted(batch, key=itemgetter(3)):
                if id_ == '0':
                    print(f'\n{id_}:\t{snt}\t{proba}\n')
                else:
                    print(f'{id_}:\t{snt}\t{proba}')
            print('*************************************************************************************')
    elif args.mode == 'sample':
        assert batch_size > args.choice_num
        np.random.seed(123)
        random.seed(123)
        clf = joblib.load(args.clf_name)
        probas = clf.predict_proba(X)[:, 1]
        corpus_batch_itr = zip(*[iter(corpus)] * batch_size)
        probas_batch_itr = zip(*[iter(probas)] * batch_size)

        def indice_generator(probs):
            probs_norm = probs / sum(probs)
            indice = []
            while len(indice) < args.choice_num-2:
                index = np.argmax(np.random.multinomial(1, probs_norm))
                if index not in indice:
                    indice.append(index)
            return indice

        if args.format == 'default':
            header = 'id\tref_loc\tbest_loc\t記事\t見出し1\t見出し2\t見出し3\t見出し4\t見出し5\t見出し6'
        elif args.format == 'yahoo':
            header = '設問ID(半角英数字20文字以内)\tチェック設問有無(0:無 1:有)\tチェック設問の解答(F04用)\t' \
                     'F01:ラベル\tF02:ラベル\tF03:ラベル\tF04:チェックボックス\tF05:ラベル'
        file_path = Path(args.tsv_file_path)
        loc_file_path = file_path.parent / ('location_' + file_path.name)
        with file_path.open('w') as wf, loc_file_path.open('w') as locf:
            print(header, file=wf)
            print('question_id\tref_id\tbest_id', file=locf)
            for i, (snt_b, probas, src) in enumerate(zip(corpus_batch_itr, probas_batch_itr, sources)):
                reference = snt_b[0]
                best = snt_b[1]
                snt_b = np.array(snt_b[2:])
                ixs = indice_generator(probas[2:])
                picked = snt_b[ixs]
                if reference == best:
                    defaults = [reference]
                    unique_num = args.choice_num - 1
                else:
                    defaults = [reference, best]
                    unique_num = args.choice_num
                while len(set(list(picked) + defaults)) < unique_num:
                    ixs = indice_generator(probas[2:])
                    picked = snt_b[ixs]
                headlines = list(picked)
                ref_loc, best_loc = random.sample(range(args.choice_num), k=2)
                if ref_loc < best_loc:
                    headlines.insert(ref_loc, reference)
                    headlines.insert(best_loc, best)
                else:
                    headlines.insert(best_loc, best)
                    headlines.insert(ref_loc, reference)
                if args.format == 'default':
                    joined_headlines = '\t'.join(headlines)
                    print(f'{i}\t{ref_loc}\t{best_loc}\t{src}\t{joined_headlines}', file=wf)
                elif args.format == 'yahoo':
                    joined_headlines = "@".join(headlines) + '@該当なし'
                    print(f'{i}\t0\t\t記事の内容から逸脱していない見出しを全てチェックしてください。\t記事\t{src}\t'
                          f'{joined_headlines}\t記事の内容から逸脱していない見出しを全てチェックしてください。', file=wf)
                print(f'{i}\t{ref_loc}\t{best_loc}', file=locf)
                if args.verbose:
                    print(f'source: {src}')
                    print(f'ref loc: {ref_loc}, best loc: {best_loc}')
                    print(*headlines, sep='\n')
                    print('*********************')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('ppl_file', nargs='?', default=None)
    parser.add_argument('-n', '--clf-name', default='model/clf.pkl')
    parser.add_argument('--format', choices=['default', 'yahoo'], default='default')
    parser.add_argument('--tsv-file-path', default='data/out.tsv')
    parser.add_argument('--choice-num', type=int, default=6)
    parser.add_argument('-bow', '--bag-of-words', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mode', choices=['train', 'eval', 'sample', 'random', 'param'])
    args = parser.parse_args()

    if args.mode == 'param':
        clf = joblib.load(args.clf_name)
        print(clf.coef_)
    elif args.mode == 'random':
        with open(args.json_file) as jsonf:
            for line in tqdm(jsonf):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
            print(d['source'])
            print(*random.sample([hypo['text'] for hypo in d['hypos']], args.choice_num), sep='\n')
            print('*************************************************************************************')
    else:
        main(args)
