import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
