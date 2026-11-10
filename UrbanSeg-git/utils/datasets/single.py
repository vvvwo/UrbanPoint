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
from utils.preprocess.preprocess_single import process_scene
import time


class SingleUnlabeledDataset(Dataset):
    def __init__(self, cfg, path_list=None):
        super(SingleUnlabeledDataset, self).__init__()
        transform = cfg["transform"]

        self.transform = AugmentationBuilder(transform)

        # follow RandLa use random sampling choose min possibility point to get data
        self.min_possibility = []
        self.possibility = []

        assert path_list is not None
        self.data_list_path = path_list
        self.data_list = []
        self.tree_list = []

        # load data and kdtree
        for path in self.data_list_path:
            data = process_scene(path)
            self.data_list.append(data)

    def get_data(self, idx):
        # update possibility
        data = self.data_list[idx % len(self.data_list)]
        name = basename(self.data_list_path[idx % len(self.data_list)])
        data_dict = {"coord": data[0].astype(np.float32),
                     "color": data[1].astype(np.float32),
                     'height': data[2].reshape(-1, 1).astype(np.float32),
                     "name": name}

        return data_dict

    def __getitem__(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def __len__(self):
        # return len(self.data_list_path)
        return len(self.data_list)
