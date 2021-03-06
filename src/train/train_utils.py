from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from data_utils.datasets import WindowDataset
from data_utils.load_dataset import load_train_data, load_dev_data, load_test_data
from data_utils.preprocess import preprocess
from data_utils.tokenizer import text_to_tokens


def collate(items):
    x_batch = torch.stack([item['x_tokens'] for item in items])
    y_batch = torch.stack([item['y_tokens'] for item in items])
    return x_batch, y_batch


def prepare_dataloader(lines: List[str], window_size: int = 512, window_n=2, train: bool = True) -> DataLoader:
    preprocessed = preprocess(lines)
    xy_tokens = text_to_tokens(preprocessed, window_size=window_size, window_n=window_n)
    print(f'Total {len(xy_tokens[0])} tokens')
    ds = WindowDataset(xy_tokens[0], xy_tokens[1], window_size=window_size, step=window_size // window_n)
    print(f'Dataset length: {len(ds)}')
    
    if train:
        dl = DataLoader(ds, shuffle=True, batch_size=8, num_workers=2, drop_last=False, collate_fn=collate)
    else:
        dl = DataLoader(ds, shuffle=False, batch_size=8, num_workers=2, drop_last=False, collate_fn=collate)
    print(f'Total {len(iter(dl))} batches')
    return dl


def get_train_dev() -> Tuple[DataLoader, DataLoader]:
    train_lines = load_train_data()
    print('Processing train data')
    train_dl = prepare_dataloader(train_lines, train=True)

    dev_lines = load_dev_data()
    print('Processing dev data')
    dev_dl = prepare_dataloader(dev_lines, train=False)
    return train_dl, dev_dl


def get_test(window_n=2) -> DataLoader:
    test_lines = load_test_data()
    print('Processing test data')
    test_dl = prepare_dataloader(test_lines, window_n=window_n, train=False)
    return test_dl