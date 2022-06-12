from transformers import AutoTokenizer
from utils.mappings import NAME2ID, PUNCT2NAME
import re

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

PUNCT2ID = {p: NAME2ID[n] for p, n in PUNCT2NAME.items()}
PUNCT2BERT_ID = {p: tokenizer(p)["input_ids"][1] for p in PUNCT2ID.keys()}
ID2BERT_ID = {PUNCT2ID[p]: i for p, i in PUNCT2BERT_ID.items()}
padding_el = NAME2ID["_PAD"]
empty = NAME2ID["_EMPTY"]


def normalize_x(text):
    t = re.sub(r'[^\w\s\.,\?]+', '', text)
    return re.sub(r'[\.,\?]+', ' ', t)


def normalize_y(text):
    return re.sub(r'[^\w\s\.,\?]+', '', text)


def construct_label(text):
    tokens = tokenizer.tokenize(normalize_y(text))
    y = [padding_el]
    for i in range(len(tokens)):
        token = tokens[i]
        if i < len(tokens) - 1 and tokens[i + 1].startswith("##"):
            y.append(padding_el)
        elif token in PUNCT2ID:
            if len(y) > 0:
                y.pop()
                y.append(PUNCT2ID[token])
        else:
            y.append(empty)

    y.append(padding_el)
    return y


def text_to_tokens(text, window_size):
    bert_input = normalize_x(text)
    encoded_input = tokenizer(bert_input)
    # tokens = tokenizer.tokenize(bert_input)
    label = construct_label(text)

    padding_size = window_size // 2
    padding_prefix = [padding_el for _ in range(padding_size)]

    return padding_prefix + encoded_input["input_ids"] + padding_prefix, padding_prefix + label + padding_prefix
    # return padding_prefix + [101] + tokens + [102] + padding_prefix, padding_prefix + label + padding_prefix


def tokens_to_text(x, y):
    n = len(x)
    x = [e for e in x if e != padding_el]
    window_size = n - len(x)

    x = x[1: -1]
    y = y[window_size // 2 + 1: -window_size // 2 - 1]
    res_tokens = []

    for i in range(len(x)):
        res_tokens.append(x[i])
        p = y[i]
        if p in ID2BERT_ID:
            res_tokens.append(ID2BERT_ID[p])

    return tokenizer.decode(res_tokens)


if __name__ == '__main__':
    x, y = text_to_tokens(
        " What? this is telling us, really,"
        "  is that we might. be. thinking. of ourselves and of other people in terms of two selves?",
        30
    )
    print(x, y)
    print(tokens_to_text(x, y))
