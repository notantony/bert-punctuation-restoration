from typing import Any, Optional, Union

import torch


class WindowDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        x_tokens: list,
        y_tokens: list,
        window_size: int = 512,
        step: Union[int, float] = 0.5,
        ):

        super().__init__()

        assert len(x_tokens) == len(y_tokens), f'Token lists must be same the length: x: {len(x_tokens)}, y: {len(y_tokens)}'

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


    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.window_size

        return {
            'x_tokens': torch.tensor(self.x_tokens[start_idx: end_idx], dtype=torch.long),
            'y_tokens': torch.tensor(self.y_tokens[start_idx: end_idx], dtype=torch.long),
        }

    def __len__(self):
        return (len(self.x_tokens) - (self.window_size - self.step)) // self.step
