from itertools import tee, zip_longest, islice

from gensim.models.keyedvectors import KeyedVectors


def create_vocabulary(path):
    print('Started loading vectors')
    w2v = KeyedVectors.load_word2vec_format(path, binary=True, encoding='utf-8')
    print('Loaded.')
    print(len(w2v.wv.vocab))


def pairwise(iterable, n=2):
    tuples = zip_longest(*(islice(it, pos, None) for pos, it in enumerate(tee(iterable, n))))
    print(list(tuples))
    return [''.join(ngram_pair) for ngram_pair in tuples]


def ngramify(word, n=2):
    chars = list(word)
    return pairwise(chars, n)


if __name__ == '__main__':
    ngrams = ngramify('something')
    print(list(ngrams))
