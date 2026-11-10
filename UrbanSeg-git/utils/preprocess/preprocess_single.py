import os
import argparse
import glob
import numpy as np
import pickle
import voxel_height
from itertools import repeat
from utils.helper_ply import write_ply, read_ply
from utils.misc import fnv_hash, grid_sub_sampling
import pandas as pd
from os.path import join, basename


def process_scene(scene_path, grid_size=1, keep_points_num=3, pillar_size=0.3):
    print("Parsing: {}".format(basename(scene_path)))
    format = basename(scene_path)[-3:]

    if format == "txt":
        obj = np.loadtxt(scene_path)
        coords = obj[:, :3].astype(np.float32)
        colors = obj[:, 3:6].astype(np.float32)
    elif format == "ply":
        data = read_ply(scene_path)
        coords = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
        colors = np.vstack((data['red'], data['green'], data['blue'])).T.astype(np.uint8)
    else:
        raise IOError("Please enter the correct file format(.txt,.ply)")
    data = None

    # grid normalization sampling
    grid_coords = np.floor(coords / grid_size)
    idx_hash = fnv_hash(grid_coords)
    v_count = np.unique(idx_hash).shape[0]
    idx = np.random.choice(coords.shape[0], min(v_count * keep_points_num, coords.shape[0]))
    coords = coords[idx]
    colors = colors[idx]

    height = voxel_height.compute(coords, pillar_size)

    return coords, colors, height
