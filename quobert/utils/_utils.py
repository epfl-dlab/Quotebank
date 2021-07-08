import random
from typing import List, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def set_seed(seed: int):
    """
    Fix all possible seeds for reproducibility
    
    Args:
        seed (int): number used to set the seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore


def get_device() -> torch.device:
    """ Check if CUDA is available """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_list(tensor: torch.Tensor) -> List[T]:
    return tensor.detach().cpu().tolist()
