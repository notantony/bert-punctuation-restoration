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


def predict(model, dl, device):
    model.to(device)
    model.eval()

    xs = []
    ys = []
    previous_suffix = None
    with torch.no_grad():
        for x_batch, _ in tqdm(dl):
            x_batch = x_batch.to(device)
            out = F.softmax(model(x_batch), dim=2)

            out = out.cpu().detach().numpy()
            x_batch = x_batch.cpu().detach().numpy()

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

                previous_suffix = b[window_half:]

    # Remove padding elements in the suffix
    xs = [x for x in xs if x != NAME2ID["_PAD"]]
    ys = ys[:len(xs)]
    print(len(xs))
    print(len(ys))
    return xs, ys


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
    print(xs)
    print(ys)
    print(len(xs))
    print(len(ys))
