import os
import glob
import numpy as np
import torch
import pickle
from copy import deepcopy
from os.path import basename
from torch.utils.data import Dataset
from collections.abc import Sequence
from utils.augmentations import AugmentationBuilder
from utils.helper_ply import read_ply
import time


class UrbanBISDataset(Dataset):
    def __init__(self, cfg, batch_size):
        super(UrbanBISDataset, self).__init__()
        data_root = cfg["data_root"]
        transform = cfg["transform"]

        self.data_root = data_root.replace('/', '\\')
        self.transform = AugmentationBuilder(transform)
        self.step = cfg["step"]
        self.batch_size = batch_size

        # follow RandLa use random sampling choose min possibility point to get data
        self.min_possibility = []
        self.possibility = []

        self.data_list_path = glob.glob(os.path.join(data_root, "*.ply"))
        self.data_list = []
        self.tree_list = []

        # load data and kdtree
        for path in self.data_list_path:
            data = read_ply(path)
            self.data_list.append(data)

    def get_data(self, idx):
        # update possibility
        data = self.data_list[idx % len(self.data_list)]
        name = basename(self.data_list_path[idx % len(self.data_list)])
        data_dict = {"coord": np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32),
                     "color": np.vstack((data['red'], data['green'], data['blue'])).T.astype(np.float32),
                     "segment": data['class'].astype(np.int32),
                     'height': data['height'].reshape(-1, 1).astype(np.float32),
                     "name": name}

        return data_dict

    def __getitem__(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def __len__(self):
        # return len(self.data_list_path)
        return self.step * self.batch_size
