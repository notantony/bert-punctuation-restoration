import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.datasets import WindowDataset
from data_utils.download_data import download_gdown
from data_utils.load_dataset import load_dev_data
from data_utils.predict import predict
from data_utils.preprocess import preprocess, text2tokens
from data_utils.tokenizer import text_to_tokens, tokens_to_text
from models import load_bert_classifier
from utils.mappings import NAME2ID


def collate(items):
    x_batch = torch.stack([item['x_tokens'] for item in items])
    y_batch = torch.stack([item['y_tokens'] for item in items])
    return x_batch, y_batch


def restore(text, model, device):
    preprocessed_text = preprocess([text], True)
    tokens = text_to_tokens(preprocessed_text, window_size=512)
    dev_ds = WindowDataset(tokens[0], tokens[1], window_size=512)
    dl = DataLoader(dev_ds, shuffle=False, batch_size=8, num_workers=1, drop_last=False, collate_fn=collate)

    xs, ys = predict(model, dl, device)

    return tokens_to_text(xs, ys)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_bert_classifier()

    download_gdown()

    sample_text = "sample text another sentence where is my otter dash 4 letters and print pieceofunbrokentext"
    text = restore(sample_text, model, device)
    print(text)
