import argparse
import json
from operator import itemgetter
from pathlib import Path
import random

import numpy as np
from nltk.util import ngrams
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from tqdm import tqdm


def ngram_score(source, sentences):
    """
    calculate scores of ngrams and normalized ngrams for each sentence against the source.
    :param source: source text such as articles
    :param sentences: list of texts such as headlines
    :return: scores of unigram, bigram, trigram and the same ngrams normalized by length
    """

    def _score(src, snts, n):
        src_gram = list(ngrams(src, n))
        snt_grams = [ngrams(snt, n) for snt in snts]
        return [len([True for g in snt_gram if g in src_gram]) for snt_gram in snt_grams]
    uni_scores = _score(source, sentences, 1)
    bi_scores = _score(source, sentences, 2)
    tri_scores = _score(source, sentences, 3)
    uni_scores_norm = [a/len(b) for a, b in zip(uni_scores, sentences)]
    bi_scores_norm = [a/len(b) for a, b in zip(bi_scores, sentences)]
    tri_scores_norm = [a/len(b) for a, b in zip(tri_scores, sentences)]
    return list(zip(uni_scores, bi_scores, tri_scores, uni_scores_norm, bi_scores_norm, tri_scores_norm))


def snt2rawtext(s, replace_underscore=True):
    """
    convert Japanese sentence tokenized by sentencepiece to the raw text.
    this function also replace underscores to spaces. This is for JNC and JAMUL corpus.

    :param s: Japanese sentence tokenized by sentencepiece
    :param replace_underscore: replace underscores to spaces if the flag is True
    :return: raw text
    """

    s = s.replace(' ', '')
    raw_text = s[1:]
    if replace_underscore:
        raw_text = raw_text.replace('_', '　')
    return raw_text


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
            source = snt2rawtext(d['source'])
            sources.append(source)
            sentences = [snt2rawtext(hypo['text']) for hypo in d['hypos']]
            corpus.extend(sentences)
            gen_scores.extend([[hypo['score']] for hypo in d['hypos']])
            gram_scores.extend(ngram_score(source, sentences))
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
        with open(f'{args.tsv_file_name}.tsv', 'w') as wf, open(f'{args.tsv_file_name}_id.tsv', 'w') as idf:
            print('設問ID(半角英数字20文字以内)\tチェック設問有無(0:無 1:有)\tチェック設問の解答(F04用)\t'
                  'F01:ラベル\tF02:ラベル\tF03:ラベル\tF04:チェックボックス\tF05:ラベル', file=wf)
            print('question_id\tref_id\tbest_id', file=idf)
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
                ref_id, best_id = random.sample(range(args.choice_num), k=2)
                if ref_id < best_id:
                    headlines.insert(ref_id, reference)
                    headlines.insert(best_id, best)
                else:
                    headlines.insert(best_id, best)
                    headlines.insert(ref_id, reference)
                joined_headlines = "@".join(headlines) + '@該当なし'
                print(f'{i}\t0\t\t記事の内容から逸脱していない見出しを全てチェックしてください。\t'
                      f'記事\t{src}\t{joined_headlines}\t記事の内容から逸脱していない見出しを全てチェックしてください。', file=wf)
                print(f'{i}\t{ref_id}\t{best_id}', file=idf)
                if args.verbose:
                    print(f'source: {src}')
                    print(f'ref id: {ref_id}, best id: {best_id}')
                    print(*headlines, sep='\n')
                    print('*************************************************************************************')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('ppl_file', nargs='?', default=None)
    parser.add_argument('-n', '--clf-name', default='model/clf.pkl')
    parser.add_argument('--format', choices=['default', 'yahoo'], default='default')
    parser.add_argument('--tsv-file-name', default='data/yahoo_template81')
    parser.add_argument('--choice-num', type=int, default=6)
    parser.add_argument('-bow', '--bag-of-words', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-e", "--eval", action="store_true")
    group.add_argument("-s", "--sample", action="store_true")
    group.add_argument("-r", "--random", action="store_true")
    group.add_argument("-p", "--param", action="store_true")
    args = parser.parse_args()

    if args.param:
        clf = joblib.load(args.clf_name)
        print(clf.coef_)
    elif args.random:
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
