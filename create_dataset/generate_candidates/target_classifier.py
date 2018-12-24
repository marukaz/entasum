import argparse
import json
from operator import itemgetter
from pathlib import Path

import numpy as np
from nltk.util import ngrams
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from tqdm import tqdm


def ngram_score(source, sentences):
    source_concat = source.replace(' ', '')

    def _score(src, snts, n):
        src_gram = list(ngrams(src, n))
        snt_grams = [ngrams(snt.replace(' ', ''), n) for snt in snts]
        for snt_gram in snt_grams:
            yield len([True for g in snt_gram if g in src_gram])
    uni_scores = list(_score(source_concat, sentences, 1))
    bi_scores = list(_score(source_concat, sentences, 2))
    tri_scores = list(_score(source_concat, sentences, 3))
    uni_scores_div = [a/len(b) for a, b in zip(uni_scores, sentences)]
    bi_scores_div = [a/len(b) for a, b in zip(bi_scores, sentences)]
    tri_scores_div = [a/len(b) for a, b in zip(tri_scores, sentences)]
    return list(zip(uni_scores, bi_scores, tri_scores, uni_scores_div, bi_scores_div, tri_scores_div))


def main(args):
    corpus = []
    sources = []
    gen_scores = []
    gram_scores = []
    features = []
    with open(args.json_file) as jsonf:
        for line in tqdm(jsonf):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sources.append(d['source'])
            sentences = [hypo['text'] for hypo in d['hypos']]
            corpus.extend(sentences)
            gen_scores.extend([[hypo['score']] for hypo in d['hypos']])
            gram_scores.extend(ngram_score(d['source'], sentences))
    batch_size = len(d['hypos'])
    features.append((csr_matrix(gen_scores)))
    features.append((csr_matrix(gram_scores)))

    if args.bag_of_words:
        if args.train:
            cv = CountVectorizer()
            bag_of_words = cv.fit_transform(corpus)
            joblib.dump(cv, 'cv_' + args.clf_name)
        elif args.eval:
            cv = joblib.load('cv_' + args.clf_name)
            bag_of_words = cv.transform(corpus)
        features.append(bag_of_words)

    if args.ppl_file is not None:
        ppl_scores = []
        with open(args.ppl_file) as pplf:
            for line in pplf:
                ppl_scores.append([float(line.split('\t')[2])])
        features.append(ppl_scores)

    X = hstack(features)
    y = [0]*len(gen_scores)
    for i in range(len(y)):
        if i % batch_size == 0:
            y[i] = 1
    if args.train:
        clf = LogisticRegression()
        print('start to learn')
        clf.fit(X, y)
        print(clf.score(X, y))
        joblib.dump(clf, args.clf_name)
    elif args.eval:
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
    elif args.sample:
        assert batch_size > args.choice_num
        clf = joblib.load(args.clf_name)
        probas = clf.predict_proba(X)[:, 1]
        corpus_batch_itr = zip(*[iter(corpus)] * batch_size)
        probas_batch_itr = zip(*[iter(probas)] * batch_size)

        def indice_generator(probas_itr):
            for probs in probas_itr:
                probs_norm = probs / sum(probs)
                indice = []
                while len(indice) < args.choice_num:
                    index = np.argmax(np.random.multinomial(1, probs_norm))
                    if index not in indice:
                        indice.append(index)
                yield indice
        for snt_b, ixs, src in zip(corpus_batch_itr, indice_generator(probas_batch_itr), sources):
            print(f'source: {src}')
            snt_b = np.array(snt_b)
            choices = snt_b[ixs]
            print(*choices, sep='\n')
            print('*************************************************************************************')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('ppl_file', nargs='?', default=None)
    parser.add_argument('-n', '--clf-name', default='model/clf.pkl')
    parser.add_argument('--choice-num', type=int, default=5)
    parser.add_argument('-bow', '--bag-of-words', action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-e", "--eval", action="store_true")
    group.add_argument("-s", "--sample", action="store_true")
    group.add_argument("-p", "--param", action="store_true")
    args = parser.parse_args()


    if args.param:
        clf = joblib.load(args.clf_name)
        print(clf.coef_)
    else:
        main(args)
