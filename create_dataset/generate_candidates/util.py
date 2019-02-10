import json

from nltk.util import ngrams
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
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


def feature_extractor(args):
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
    return X, batch_size, gen_scores, corpus, sources
