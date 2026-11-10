import os
import argparse
import glob
import numpy as np
import pickle
import voxel_height
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from utils.helper_ply import write_ply, read_ply
from utils.misc import fnv_hash, grid_sub_sampling
import pandas as pd
from os.path import join, basename


def parse_scene(scene_path, output_path, grid_size, keep_points_num, pillar_size):
    print("Parsing: {}".format(basename(output_path)))

    obj = np.loadtxt(scene_path)
    coords = obj[:, :3].astype(np.float32)
    colors = obj[:, 3:6].astype(np.float32)
    labels = obj[:, 6].astype(np.uint8)
    labels = labels.reshape([-1, 1])
    data = None

    # grid normalization sampling
    grid_coords = np.floor(coords / grid_size)
    idx_hash = fnv_hash(grid_coords)
    v_count = np.unique(idx_hash).shape[0]
    idx = np.random.choice(coords.shape[0], min(v_count * keep_points_num, coords.shape[0]))
    coords = coords[idx]
    colors = colors[idx]
    labels = labels[idx]

    height = voxel_height.compute(coords, pillar_size)

    save_path = output_path
    write_ply(save_path, [coords, colors, labels, height], ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'height'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name_list", nargs="+", default=['Lihu', 'Longhua', 'Qingdao', 'Wuhu', 'Yingrenshi', 'Yuehai'],
                        help="Scene need to process.", )
    parser.add_argument("--dataset_root", required=True, help="Path to UrbanBIS(Absolute path)")
    parser.add_argument("--output_root", default="data/UrbanBIS",
                        help="Output path where splits folders will be located", )
    parser.add_argument("--pillar_size", default=0.5, type=float,
                        help="The voxel height calculates the size of the 2D voxel selected by the module.")
    parser.add_argument("--grid_size", default=1, type=int, help="Voxel normalization The selected voxel size.")
    parser.add_argument("--num_workers", default=6, type=int, help="Num workers for preprocessing.")
    parser.add_argument("--keep_points_num", default=3, type=int,
                        help="Num of retention points per voxel after voxel normalization.")
    args = parser.parse_args()

    # preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(
        max_workers=args.num_workers
    )

    # Add Val and Train data path
    data_path = []
    output_path = []
    for split in ['val', 'train']:
        output = join(args.output_root, split)
        os.makedirs(output) if not os.path.exists(output) else None

        for scene in args.scene_name_list:
            path_list = glob.glob(join(args.dataset_root, "train", scene, split, "*.txt"))
            data_path += path_list
            output_path += [join(output, f"{scene}_{basename(scene_path)[:-4]}.ply") for scene_path in path_list]

    # Add Test date path
    output = join(args.output_root, "test")
    os.makedirs(output) if not os.path.exists(output) else None

    for scene in args.scene_name_list:
        path_list = glob.glob(join(args.dataset_root, "test", scene, "*.txt"))
        data_path += path_list
        output_path += [join(output, f"{scene}_{basename(scene_path)[:-4]}.ply") for scene_path in path_list]

    _ = list(
        pool.map(
            parse_scene,
            data_path,
            output_path,
            repeat(args.grid_size),
            repeat(args.keep_points_num),
            repeat(args.pillar_size),
        )
    )


if __name__ == "__main__":
    main()
