from typing import Any, Optional, Union

import torch


class WindowDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        x_tokens: list,
        y_tokens: list,
        window_size: int = 512,
        step: Union[int, float] = 0.5,
        pad_size: Optinal[int] = None,
        pad_item: Any = None,
        ):

        super().__init__()

        if isinstance(step, float):
            step = int(window_size * step)
        elif isinstance(step, int):
            pass
        else:
            raise ValueError(f"Unexpected `step` parameter type: {type(step)}")

        self.step = step
        self.window_size = window_size
        self.x_tokens = x_tokens
        self.y_tokens = y_tokens

        remaining_shift = window_size % step
        self.pad_size = window_size - remaining_shift

    def __getitem__(self, idx):
        start_idx = idx * self.window_size + self.

        return {
            'x_tokens': self.x_tokens[idx * self.window_size + ], 
            'y_tokens': self.y_tokens[idx]
        }

    def __len__(self):
        return len(self.x_sents)
