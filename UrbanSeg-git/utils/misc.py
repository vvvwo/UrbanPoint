import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import os
from datetime import datetime


def fnv_hash(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def grid_sub_sampling(points, grid_size=0.1):
    grid_coords = np.floor(points / grid_size)
    idx_hash = fnv_hash(grid_coords)
    idx_sort = np.argsort(idx_hash)
    key_sort = idx_hash[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]
    return idx_unique


def offset2batch(offset):
    return (
        torch.cat(
            [
                torch.tensor([i] * (o - offset[i - 1]))
                if i > 0
                else torch.tensor([i] * o)
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to(offset.device)
    )


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_random_seed():
    seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
    )
    return seed
