# UrbanSeg

Source code for urban point cloud semantic segmentation, including the UrbanSeg model, UrbanBIS data loading and preprocessing, training, and inference on individual scenes.

**This repository contains source code only. Datasets, pretrained weights, training checkpoints, predicted point clouds, logs, and intermediate results are not included.** Inference requires a compatible checkpoint that you provide. You can also train a model using your own prepared data.

## Repository structure

```text
main.py                         Command-line entry point for training and inference
trainer.py / tester.py          Training and inference implementations
configs/urban_seg.py            Inference configuration for individual scenes
configs/urban_seg_train.py      UrbanBIS training configuration
utils/model/                   Model implementation
utils/datasets/                Datasets and batch collation
utils/preprocess/              UrbanBIS, SensatUrban, and individual-scene preprocessing
utils/augmentations/           Data augmentation
utils/losses/                  Loss functions
utils/serialization/           Point cloud serialization
utils/cpp_wrappers/             voxel_height C++ extension and supporting source code
LabelsFileGenerate.py           Predicted label export and semantic coloring
plyRGBcorrect.py               PLY RGB conversion utility
```

## Environment setup

An NVIDIA GPU, a CUDA environment, and C++ build tools are required. The original project documented two environment configurations: Python 3.8 / PyTorch 2.1.0 / CUDA 11.8 and Python 3.10 / PyTorch 2.5.0 / CUDA 12.4. GPU execution has not been revalidated for this source-only edition. Start with an existing working environment if available. The current extension build uses `numpy.distutils`; use Python 3.8–3.10 and NumPy 1.x.

The following installation example follows the original CUDA 11.8 configuration. A fully pinned dependency lock file is not provided.

```powershell
conda create -n urbanseg python=3.8
conda activate urbanseg
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install spconv-cu118
```

The CUDA versions used by PyTorch, torch-scatter, and spconv must match. The original installation instructions also listed torchvision, torchaudio, torch-cluster, torch-sparse, and torch-geometric. The current source does not import these packages directly, so they are omitted from the minimal dependency list.

Build the height calculation extension, starting from the repository root:

```powershell
cd utils/cpp_wrappers/cpp_height
python setup.py install
python test.py
cd ../../..
```

On Windows, install Visual Studio C++ Build Tools. The existing `setup.py` retains the original GCC compiler options; MSVC may report that these options are ignored. Extension compilation and GPU compatibility must be verified in the target environment.

## Data preparation

Obtain the dataset separately. The UrbanBIS preprocessing script reads TXT files with at least seven columns per row: `x y z r g b class`. RGB values use the range 0–255, and class IDs range from 0 to 6. The script searches for input files using this layout:

```text
<dataset_root>/train/<scene>/train/*.txt
<dataset_root>/train/<scene>/val/*.txt
<dataset_root>/test/<scene>/*.txt
```

Always run preprocessing modules from the repository root to ensure that `utils` can be imported:

```powershell
python -m utils.preprocess.preprocess_urbanbis --dataset_root "D:/datasets/UrbanBIS" --output_root "data/UrbanBIS" --pillar_size 0.5 --keep_points_num 3
```

The default scenes are Lihu, Longhua, Qingdao, Wuhu, Yingrenshi, and Yuehai. Override them with, for example, `--scene_name_list Lihu Longhua`. Outputs are written to `data/UrbanBIS/train`, `val`, and `test`. These generated files are excluded from Git.

SensatUrban preprocessing source is also included. However, this repository currently provides neither a corresponding training configuration nor a dedicated dataset loader, so changing the data path alone does not provide complete training support.

## Training

Review the data root (`data_root`), batch size (`batch_size`), number of epochs (`epoch`), and other settings in `configs/urban_seg_train.py` to match your local data and available GPU memory. Then run the following command from the repository root:

```powershell
python main.py --mode train --exp-name urban_seg
```

Training logs and checkpoints are saved under `result/urban_seg/`. The best checkpoint is saved as `result/urban_seg/model/model_best.pth`. The current command-line interface starts training from scratch and does not expose a resume option; the original resume implementation remains in `trainer.py`.

## Inference on individual scenes

Inputs can be PLY files containing XYZ and RGB fields (`x y z red green blue`), or TXT files whose first six columns contain XYZ and RGB. Provide a checkpoint compatible with the model configuration, containing model parameters (`state_dict`) and the training epoch (`epoch`):

```powershell
python main.py --mode test --input "data/scene.ply" --weight "result/urban_seg/model/model_best.pth" --exp-name urban_seg
```

You can supply multiple files after `--input`. The default configuration is `configs/urban_seg.py`. Use `--config` to select another configuration module under `configs/`, without the `.py` extension. Predictions are saved under `result/<exp-name>/save/` and contain coordinates, original RGB values, and a `pred` field. TXT inputs also produce PLY outputs.

Preprocessing includes random sampling. Predictions correspond to the points retained after preprocessing, rather than every point in the input file. This inference workflow does not compute test-set metrics against ground-truth labels.

| Label | Class |
| --- | --- |
| 0 | Terrain |
| 1 | Vegetation |
| 2 | Water |
| 3 | Bridge |
| 4 | Vehicle |
| 5 | Boat |
| 6 | Building |

To export semantically colored point clouds and label TXT files, set `vis_path` and `output` in `LabelsFileGenerate.py`, then run the script. Likewise, configure your input and output paths before running `plyRGBcorrect.py`.

## Source-only release notes

This edition retains all Python and C++ source code, consolidates the configuration directories, and adds a parameterized command-line entry point. It also fixes preprocessing scripts overriding user-supplied dataset paths, configuration module path separators, and inference dropping the final incomplete batch. The original working directory is unchanged.

The `.gitignore` file excludes data, weights, predictions, logs, caches, compiled artifacts, and IDE files. Avoid using `git add -f` to include these files. Review staged changes with `git status --short` before uploading.

## License

Original contributions to this project are licensed under the [MIT License](LICENSE), which permits use, modification, redistribution, and commercial use, provided that the copyright and permission notices are retained.

Third-party code remains subject to its original licenses; the project-level MIT license does not replace upstream licenses. Author and copyright notices have been preserved, including the MIT notice in `z_order.py` and the BSD license text in `nanoflann.hpp`. Some files contain Point Transformer V3 and other upstream author notices. Their complete provenance and licensing have not yet been verified; the corresponding upstream license texts and required notices must be added before public release.

This edition has been checked for Python syntax, command-line entry point behavior, and release file contents only. C++ compilation, GPU training, and inference have not been validated.
