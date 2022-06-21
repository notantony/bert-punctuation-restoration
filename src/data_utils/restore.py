import torch
import re
from torch.utils.data import DataLoader

from data_utils.datasets import WindowDataset
from data_utils.download_data import download_gdown
from data_utils.predict import predict
from data_utils.preprocess import preprocess
from data_utils.tokenizer import text_to_tokens, tokens_to_text
from models import load_bert_classifier


substitutions = {
    "i": "I",
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "cantve": "can't've",
    "cause": "'cause'",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldntve": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadntve": "hadn't've",
    "hasnе": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hedve": "he'd've",
    # "he'll": "he shall / he will",
    "hellve": "he'll've",
    "hes": "he's",
    "howd": "how'd",
    "howdy": "how'd'y",
    "howll": "how'll",
    "hows": "how's",
    # "i'd": "I had / I would",
    "idve": "i'd've",

    "illve": "i'll've",
    "itdve": "it'd've",
    "itllve": "it'll've",
    "mightntve": "mightn't've",
    "mustntve": "mustn't've",
    "needntve": "needn't've",
    "oughtntve": "oughtn't've",
    "shantve": "shan't've",
    "shedve": "she'd've",
    "shellve": "she'll've",
    "theydve": "they'd've",
    "theyllve": "they'll've",
    "wedve": "we'd've",
    "wellve": "we'll've",
    "whatllve": "what'll've",
    "whollve": "who'll've",
    "wontve": "won't've",
    "wouldntve": "wouldn't've",
    "yalld": "y'all'd",
    "yalldve": "y'all'd've",
    "yallre": "y'all're",
    "yallve": "y'all've",
    "youdve": "you'd've",
    "youllve": "you'll've",
    "shouldntve": "shouldn't've",
    "thatdve": "that'd've",
    "theredve": "there'd've",

    "oclock": "o'clock",
    "maam": "ma'am",

    # "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    # "its": "it's",
    "lets": "let's",
    "maynt": "mayn't",
    "mightve": "might've",
    "mightnt": "mightn't",
    "mustve": "must've",
    "mustnt": "mustn't",
    "neednt": "needn't",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    # "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "sove": "so've",
    # "sos": "so's",
    "thatd": "that'd",
    "thats": "that's",
    "thered": "there'd",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "tove": "to've",
    "wasnt": "wasn't",
    "wed": "we'd",
    # "well": "we'll",
    # "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whenve": "when've",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whys": "why's",
    "whyve": "why've",
    "willve": "will've",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}


def collate(items):
    x_batch = torch.stack([item['x_tokens'] for item in items])
    y_batch = torch.stack([item['y_tokens'] for item in items])
    return x_batch, y_batch


def prettify_text(text):
    if len(text) == 0:
        return text

    substituted_words_text = text
    for w, s in substitutions.items():
        substituted_words_text = re.sub(
            r'\b' + w + r'\b',
            s,
            substituted_words_text
        )

    uppercase_letters_text = re.sub(
        r'(^|[.?!])\s*([a-zA-Z])',
        lambda p: p.group(0).upper(),
        substituted_words_text
    )

    return uppercase_letters_text


def restore(text, model, device):
    preprocessed_text = preprocess([text], True)
    tokens = text_to_tokens(preprocessed_text, window_size=512)
    dev_ds = WindowDataset(tokens[0], tokens[1], window_size=512)
    dl = DataLoader(dev_ds, shuffle=False, batch_size=8, num_workers=1, drop_last=False, collate_fn=collate)

    xs, ys = predict(model, dl, device)
    restored_text = tokens_to_text(xs, ys)

    return prettify_text(restored_text)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_bert_classifier()

    download_gdown()

    sample_text = "sample text another sentence where is my otter dash 4 letters and print pieceofunbrokentext I dont want a cake it'll be alright"
    text = restore(sample_text, model, device)
    print(text)
