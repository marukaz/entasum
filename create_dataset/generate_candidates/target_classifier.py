import argparse
import json

import numpy as np
from nltk.util import ngrams
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


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
    with open(args.file) as f:
        for line in f:
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
    cv = CountVectorizer()
    bag_of_words = cv.fit_transform(corpus)

    X = hstack((csr_matrix(gen_scores), csr_matrix(gram_scores), bag_of_words))
    y = [0]*len(gen_scores)
    for i in range(len(y)):
        if i % batch_size == 0:
            y[i] = 1
    if args.train:
        clf = LogisticRegression()
        print('start to learn')
        clf.fit(X, y)
        print(clf.score(X, y))
        joblib.dump(clf, 'clf.pkl')
    elif args.eval:
        clf = joblib.load('clf.pkl')
        probas = clf.predict_proba(X)
        corpus_batch_itr = zip(*[iter(corpus)] * batch_size)
        probas_batch_itr = zip(*[iter(probas)] * batch_size)
        for snt_b, proba_b, src in zip(corpus_batch_itr, probas_batch_itr, sources):
            print(f'source: {src}')
            batch = np.column_stack(snt_b, proba_b)
            for snt, proba in sorted(batch, key=lambda x: x[1], reverse=True):
                print(f'{snt}\t{proba}')
            print('*************************************************************************************')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-e", "--eval", action="store_true")
    args = parser.parse_args()
    main(args)
