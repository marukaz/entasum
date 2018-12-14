import argparse
import json

import numpy as np
from nltk.util import ngrams
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


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
    gen_scores = []
    gram_scores = []
    with open(args.file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sentences = [hypo['text'] for hypo in d['hypos']]
            gen_scores.extend([hypo['score'] for hypo in d['hypos']])
            gram_scores.extend(ngram_score(d['source'], sentences))
            corpus.extend(sentences)
    batch_size = len(d['hypos'])
    cv = CountVectorizer()
    bag_of_words = cv.fit_transform(corpus)

    X = hstack((csr_matrix(gen_scores), csr_matrix(gram_scores), bag_of_words))
    y = [0]*len(X)
    for i in range(len(y)):
        if i % batch_size == 0:
            y[i] = 1
    clf = LogisticRegression()
    print('start to learn')
    clf.fit(X, y)
    print(clf.score(X, y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    main(args)
