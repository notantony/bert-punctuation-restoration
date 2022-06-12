from typing import List
from utils.mappings import *


def drop_unknown(token):
    if re.match(r'[^\w\d]+', token):
        if token not in PUNCT2NAME:
            dropped_tokens.update([token])
        else:
            used_tokens.update([PUNCT2NAME[token]])
            yield PUNCT2NAME[token]
    else:
        yield token


def preprocess_punct(token):
    if token == '!' or token == ';':
        yield '.'
    elif token == '--':
        yield ','
    elif token == ',--':
        yield ','
    elif token == '...':
        yield '.'
    elif re.match(r'[^\w\d]+', token):
        for c in token:
            if token in PUNCT2NAME:
                yield token
    else:
        yield token


def text2tokens(text: str) -> List[str]:
    print('Preprocessing text')
    word_tokenizer = RegexpTokenizer(r'[\w\-\']+|[^\w\s]+')

    dropped_tokens = Counter()
    used_tokens = Counter()

    tokens = word_tokenizer.tokenize(text)
    tokens = itertools.chain.from_iterable(map(preprocess_punct, tokens))
    tokens = itertools.chain.from_iterable(map(drop_unknown, tokens))
    tokens = list(tokens)

    print(f'Dropped tokens: {dropped_tokens}')
    print(f'Used punctuation: {used_tokens}')
    return tokens


def tokens2text():
    pass


def preprocess(data: List[str], lower: bool = False) -> str:
    raw_corpus = ' '.join(data)
    tokens_data = text2tokens(raw_corpus)
