import os
import argparse
import glob
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from sklearn.decomposition import PCA
from utils.helper_ply import write_ply, read_ply
import voxel_height
from utils.misc import fnv_hash, grid_sub_sampling
import pandas as pd
from os.path import join, basename


def parse_scene(data_path, output_path, grid_size, keep_points_num, pillar_size):
    name = basename(data_path)[:-4]
    print("Parsing: {}".format(name))
    ply_path = data_path

    data = read_ply(ply_path)
    coords = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
    colors = np.vstack((data['red'], data['green'], data['blue'])).T.astype(np.uint8)
    labels = data['class'].astype(np.uint8)
    data = None
    labels = labels.reshape([-1, 1])

    # voxel normalization sampling(optional)
    grid_coords = np.floor(coords / grid_size)
    idx_hash = fnv_hash(grid_coords)
    v_count = np.unique(idx_hash).shape[0]
    idx = np.random.choice(coords.shape[0], min(v_count * keep_points_num, coords.shape[0]))
    coords = coords[idx]
    colors = colors[idx]
    labels = labels[idx]

    # use PCA for align the point cloud and compute point cloud height
    height = voxel_height.compute(coords, pillar_size)

    save_path = os.path.join(output_path, name + '.ply')
    write_ply(save_path, [coords, colors, labels, height], ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'height'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to SensatUrban(Absolute path)")
    parser.add_argument("--output_root", default="data/SensatUrban",
                        help="Output path where splits folders will be located", )
    parser.add_argument("--pillar_size", default=0.5, type=float,
                        help="The voxel height calculates the size of the 2D voxel selected by the module.")
    parser.add_argument("--grid_size", default=1, type=int, help="Voxel normalization The selected voxel size.")
    parser.add_argument("--num_workers", default=2, type=int, help="Num workers for preprocessing.")
    parser.add_argument("--keep_points_num", default=3, type=int,
                        help="Num of retention points per voxel after voxel normalization.")
    args = parser.parse_args()
    test_split = ['birmingham_block_1',
                  'birmingham_block_5',
                  'cambridge_block_10',
                  'cambridge_block_7']
    val_split = ['birmingham_block_2',
                 'birmingham_block_8',
                 'cambridge_block_15',
                 'cambridge_block_22',
                 'cambridge_block_16',
                 'cambridge_block_27']
    train_split = ['birmingham_block_0',
                   'birmingham_block_10',
                   'birmingham_block_11',
                   'birmingham_block_12',
                   'birmingham_block_13',
                   'birmingham_block_3',
                   'birmingham_block_4',
                   'birmingham_block_6',
                   'birmingham_block_7',
                   'birmingham_block_9',
                   'cambridge_block_12',
                   'cambridge_block_13',
                   'cambridge_block_14',
                   'cambridge_block_17',
                   'cambridge_block_18',
                   'cambridge_block_19',
                   'cambridge_block_2',
                   'cambridge_block_20',
                   'cambridge_block_21',
                   'cambridge_block_23',
                   'cambridge_block_25',
                   'cambridge_block_26',
                   'cambridge_block_28',
                   'cambridge_block_3',
                   'cambridge_block_32',
                   'cambridge_block_33',
                   'cambridge_block_34',
                   'cambridge_block_4',
                   'cambridge_block_6',
                   'cambridge_block_8',
                   'cambridge_block_9']

    # preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(
        max_workers=args.num_workers
    )  # peak 110G memory when parsing normal.
    for mode in ['test', 'val', 'train']:
        output = join(args.output_root, mode)
        os.makedirs(output) if not os.path.exists(output) else None
        data_path_list = []
        if mode == 'test':
            for split in test_split:
                data_path_list += [(join(args.dataset_root, split + ".ply"))]
        elif mode == 'train':
            for split in train_split:
                data_path_list += [(join(args.dataset_root, split + ".ply"))]
        elif mode == "val":
            for split in val_split:
                data_path_list += [(join(args.dataset_root, split + ".ply"))]
        _ = list(
            pool.map(
                parse_scene,
                data_path_list,
                repeat(output),
                repeat(args.grid_size),
                repeat(args.keep_points_num),
                repeat(args.pillar_size),
            )
        )


if __name__ == "__main__":
    main()
