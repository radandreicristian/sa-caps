from typing import List
from nltk.util import ngrams


def to_char_ngram(x: str, max_n):
    return list( ''.join(j) for j in
            [list(ngrams(x, n=i)) for i in range(1, max_n)]
            )


if __name__ == '__main__':
    print(to_char_ngram('today', 3))
