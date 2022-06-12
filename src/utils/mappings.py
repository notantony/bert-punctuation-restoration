
def invert_mapping(mapping: dict):
    return {v: k for k, v in mapping.items()}


NAME2ID = {
    '_EMPTY': 0,
    '_PERIOD' : 1,
    '_COMMA': 2,
    '_QUESTION': 3,
    '_PAD': 4,
}

ID2NAME = invert_mapping(NAME2ID)

PUNCT2NAME = {
    ',': '_COMMA',
    '.': '_PERIOD',
    '?': '_QUESTION',
}

NAME2PUNCT = invert_mapping(PUNCT2NAME)
