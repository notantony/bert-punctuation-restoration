import itertools
import re
from collections import Counter
from typing import List, Dict, Optional, Any, Union

from nltk.tokenize.regexp import RegexpTokenizer

from utils.mappings import *


def text2tokens(text: str, apply_preprocess: bool = False) -> List[str]:
    print('Tokenizing text')
    word_tokenizer = RegexpTokenizer(r'[\w\-\']+|[^\w\s]+')

    dropped_tokens = Counter()
    used_tokens = Counter()

    def drop_quotes(text):
        symbols_cnt = Counter(text)
        dropped_tokens["'"] += symbols_cnt["'"]
        dropped_tokens['"'] += symbols_cnt['"']
        return text.replace("'", '').replace('"', '')

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
                    dropped_tokens.update([token])
        else:
            yield token

    text = drop_quotes(text)
    tokens = word_tokenizer.tokenize(text)

    if apply_preprocess:
        tokens = itertools.chain.from_iterable(map(preprocess_punct, tokens))
        tokens = itertools.chain.from_iterable(map(drop_unknown, tokens))

    tokens = list(tokens)

    print(f'Dropped tokens: {dropped_tokens}')
    print(f'Used punctuation: {used_tokens}')
    return tokens


def tokens2text(tokens: List[str], mapping: Optional[Dict[Any, str]] = None, capitalize : bool = False):
    if capitalize:
        raise NotImplementedError()

    def map_token(token):
        if mapping and (token in mapping):
            token = mapping[token]

        if token not in PUNCT2NAME:
            token = ' ' + token 

        return token

    return ''.join(map(map_token, tokens))[1:]


def preprocess(data: List[str], lower: bool = True, return_tokens: bool = False) -> Union[str, List[str]]:
    raw_corpus = ' '.join(line.rstrip('\n') for line in data)
    if lower:
        raw_corpus = raw_corpus.lower()

    tokens = text2tokens(raw_corpus, apply_preprocess=True)
    if return_tokens:
        return tokens

    text = tokens2text(tokens, NAME2PUNCT)
    return text
