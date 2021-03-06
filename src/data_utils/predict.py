import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.datasets import WindowDataset
from data_utils.download_data import download_gdown
from data_utils.load_dataset import load_dev_data
from data_utils.preprocess import preprocess
from data_utils.tokenizer import text_to_tokens
from models import load_bert_classifier
from utils.mappings import NAME2ID


def remove_suffix_padding(xs, ys, tys, return_true):
    xs = [x for x in xs if x != NAME2ID["_PAD"]]
    ys = ys[:len(xs)]
    tys = tys[:len(xs)]

    assert len(xs) == len(ys) and len(xs) == len(tys)

    if return_true:
        return xs, ys, tys
    return xs, ys


def predict_multiwindow(model, dl, device, windows_n=2, return_true=False):
    model.to(device)
    model.eval()

    xs = []
    ys = []
    y_true = []
    previous_batches = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dl):
            x_batch = x_batch.to(device)
            out = F.softmax(model(x_batch), dim=2)

            out = out.cpu().detach().numpy()
            x_batch = x_batch.cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()

            for i in range(len(out)):
                b = out[i]
                window_shift = len(b) // windows_n

                if len(previous_batches) + 1 < windows_n:
                    previous_batches.append(b)
                    previous_batches = [b[window_shift:] for b in previous_batches]
                    continue

                current_prefix = b[:window_shift]
                previous_windows = [b[:window_shift] for b in previous_batches]
                if windows_n == 1:
                    predictions = current_prefix
                else:
                    predictions = (np.sum(previous_windows, axis=0) + current_prefix) / windows_n
                predictions = np.argmax(predictions, axis=1)

                ys.extend(predictions)
                xs.extend(x_batch[i][:window_shift])
                y_true.extend(y_batch[i][:window_shift])
                previous_batches = previous_batches[1:]
                previous_batches = [b[window_shift:] for b in previous_batches]
                previous_batches.append(b[window_shift:])

    return remove_suffix_padding(xs, ys, y_true, return_true)

def predict(model, dl, device, return_true=False):
    model.to(device)
    model.eval()

    xs = []
    ys = []
    y_true = []
    previous_suffix = None
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dl):
            x_batch = x_batch.to(device)
            out = F.softmax(model(x_batch), dim=2)

            out = out.cpu().detach().numpy()
            x_batch = x_batch.cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()

            for i in range(len(out)):
                b = out[i]
                window_half = len(b) // 2

                if previous_suffix is None:
                    previous_suffix = b[window_half:]
                    continue

                current_prefix = b[:window_half]
                predictions = (previous_suffix + current_prefix) / 2
                predictions = np.argmax(predictions, axis=1)

                ys.extend(predictions)
                xs.extend(x_batch[i][:window_half])
                y_true.extend(y_batch[i][:window_half])

                previous_suffix = b[window_half:]

    return remove_suffix_padding(xs, ys, y_true, return_true)


def collate(items):
    x_batch = torch.stack([item['x_tokens'] for item in items])
    y_batch = torch.stack([item['y_tokens'] for item in items])
    return x_batch, y_batch


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_bert_classifier()

    download_gdown()
    dev_lines = load_dev_data()[:101]
    dev_preprocessed = preprocess(dev_lines)
    dev_tokens = text_to_tokens(dev_preprocessed, window_size=512)
    dev_ds = WindowDataset(dev_tokens[0], dev_tokens[1], window_size=512)
    dl = DataLoader(dev_ds, shuffle=False, batch_size=8, num_workers=1, drop_last=False, collate_fn=collate)

    xs, ys = predict(model, dl, device)
    xs2, ys2 = predict_multiwindow(model, dl, device, 2)
    print(xs)
    print(xs2)
    print(ys)
    print(ys2)
    print(len(xs))
    print(len(xs2))
    print(len(ys))
    print(len(ys2))
