import torch
import numpy as np


def decimal_to_ternary(decimal, depth=16):
    ternary = decimal.unsqueeze(-1).repeat((1, 1, depth)).to(decimal.device).to(torch.int32)
    cache = decimal.clone()
    for i in range(depth):
        ternary[:, :, depth - i - 1] = cache % 3
        cache = cache // 3
    return ternary


def ternary_to_decimal(ternary):
    decimal = torch.zeros(ternary[:, -1].shape).to(ternary.device).to(torch.int64)
    for i in range(ternary.shape[-1]):
        decimal += ternary[:, i] * (3 ** (ternary.shape[-1] - i - 1))
    return decimal


def encode(grid_coord, depth=16):
    hash_table = torch.tensor(np.load("utils\\table\\hash_table.npy")).to(grid_coord.device)
    state_table = torch.tensor(np.load("utils\\table\\state_table.npy")).to(grid_coord.device)
    ternary = decimal_to_ternary(grid_coord, depth)
    key = ternary.flatten(-2, -1).clone()
    state = torch.zeros(grid_coord.shape[0], dtype=torch.int32)
    for i in range(depth):
        # hash = table[s][x][y][z]
        code = hash_table[state, ternary[:, 2, i], ternary[:, 1, i], ternary[:, 0, i]]
        state = state_table[state, ternary[:, 2, i], ternary[:, 1, i], ternary[:, 0, i]]
        for j in range(3):
            key[:, i * 3 + 2 - j] = code % 3
            code = code // 3

    key = ternary_to_decimal(key)
    return key
