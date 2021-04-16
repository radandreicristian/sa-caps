import json
from itertools import tee, zip_longest, islice
from typing import Iterable

import regex

from src.utils.date import get_hour_min


def pairwise(iterable: Iterable,
             tuple_size: int):
    """
    Creates tuples of size tuple_size from the elements of iterable.
    :param iterable: An iterable sequence
    :param tuple_size: The size of the resulting tuples
    :return: The tuples containing tuple_size consecutive elements from iterable.
    """
    return zip_longest(*(islice(it, pos, None) for pos, it in enumerate(tee(iterable, tuple_size))))


def ngramify(word: str,
             max_ngram_size: int):
    """
    Computes the list of all character n-grams (with n < max_ngram_size) of word.
    :param word: The word to be ngramified.
    :param max_ngram_size: The maximum character ngram size.
    :return: The list of all character n-grams.
    """
    chars = list(word)
    ngram_tuples = []
    ngrams = []
    # Take all n-grams with length in [2, max_ngram_size]
    for i in range(2, max_ngram_size + 1):
        pw = list(pairwise(chars, i))
        ngram_tuples += pw
    for e in ngram_tuples:
        if None not in e:
            ngrams.append(''.join(e))
    # Append the list of characters, to avoid an unnecessary call for pairwise(_, 1).
    return list(word) + ngrams


def ngram_to_index(ngram: str,
                   n_chars: int = 26):
    """
    Return the lexicographic index of the ngram following the pattern of index - value. Example is for an
    1 - 'a'
    ...
    26 - 'z'
    27 - 'aa'
    52  - 'az'
    53  - 'ba'
    ...
    703 - 'zz'
    704 - 'aaa'
    ...
    18278 - 'zzz'
    18279 - 'aaaa'
    ...
    Etc.
    :param n_chars: The total number of characters in the language. English defaults to 26
    :param ngram: A string, representing an arbitraty n-gram.
    :return: The index, accorcing to the rule described above.
    """
    index = 0
    ord_a = ord('a')
    indices = list(map(lambda x: ord(x) - ord_a + 1, ngram))
    for i, char in enumerate(reversed(ngram)):
        index += (indices[i]) * pow(n_chars, i)
    # Todo - This needs some thorough testing:)
    return index - 1


def f7(seq):
    return list(dict.fromkeys(seq))


def compute_sparse_features(word: str,
                            n_chars: int = 26,
                            max_ngram_size: int = 3):
    """
    Creates a sparse representation of the indices of ngrams of the given word.
    :param word: A word we want to create the sparse ngrams representation for.
    :param n_chars: Number of characters in the language.
    :param max_ngram_size: The maximum size of n-grams.
    :return: A list containing the indices. The indices are the indices of the present n-grams in the sparse
    lexicographic ngram vector described by ngram_to_index.
    """
    word_lower = word.lower()
    ones = []
    n_grams = ngramify(word_lower, max_ngram_size)
    n_grams_filtered = filter(lambda x: regex.match('^[a-z]+$', x), n_grams)
    for ngram in n_grams_filtered:
        index = ngram_to_index(n_chars=n_chars, ngram=ngram)
        ones.append(index)
    return f7(ones)


if __name__ == '__main__':

    n_ch = 26
    vocab_file = '/home/rad/projects/id-sf/data/vocabs/en_vocab.txt'

    for max_ngrams in range(2, 6):
        print(f'Started computing sparse features for n_grams up to {max_ngrams} at {get_hour_min()}')
        sparse_features = {}
        f = open(vocab_file, 'r')
        count = 0
        with(open(f'/home/rad/projects/id-sf/data/embeddings/sparse/en_{max_ngrams}.json', 'w')) as out:
            for line in f:
                line = line.replace('\n', '').replace('\"', '')
                sparse_features[str(line)] = compute_sparse_features(str(line), n_ch, max_ngrams)

            # Add special cases
            sparse_features['<pad>'] = []
            sparse_features['<mask>'] = []
            sparse_features['<eof>'] = []

            json.dump(sparse_features, out)
            out.close()
        f.close()
        print(f'Finished computing sparse features for n_grams up to {max_ngrams} at {get_hour_min()}')
