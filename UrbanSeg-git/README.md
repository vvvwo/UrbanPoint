# UrbanSeg

城市点云语义分割源码，包含 UrbanSeg 模型、UrbanBIS 数据加载与预处理、训练及单场景推理。

**本仓库仅发布源码，不包含数据集、预训练权重、训练检查点、预测点云、日志或任何中间结果。** 推理需自行准备匹配的权重；也可以使用自行准备的数据训练。

## 目录

```text
main.py                         训练 / 推理命令行入口
trainer.py / tester.py          训练与推理实现
configs/urban_seg.py            单场景推理配置
configs/urban_seg_train.py      UrbanBIS 训练配置
utils/model/                   模型实现
utils/datasets/                数据集与批处理
utils/preprocess/              UrbanBIS / SensatUrban 及单场景预处理
utils/augmentations/           数据增强
utils/losses/                  损失函数
utils/serialization/           点云序列化
utils/cpp_wrappers/             voxel_height C++ 扩展及依赖源码
LabelsFileGenerate.py           预测标签导出与语义着色
plyRGBcorrect.py               PLY RGB 转换辅助脚本
```

## 环境

需要 NVIDIA GPU、CUDA 环境和 C++ 编译工具。原项目记录过 Python 3.8 / PyTorch 2.1.0 / CUDA 11.8，以及 Python 3.10 / PyTorch 2.5.0 / CUDA 12.4 两组环境；本整理版未重新验证 GPU 运行。建议先沿用已有可运行环境。现有扩展构建使用 `numpy.distutils`，请使用 Python 3.8–3.10 和 NumPy 1.x。

以下为沿用原项目 CUDA 11.8 环境的安装示例，依赖未提供完整锁定文件：

```powershell
conda create -n urbanseg python=3.8
conda activate urbanseg
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install spconv-cu118
```

PyTorch、torch-scatter、spconv 的 CUDA 版本需要匹配。原安装说明还列过 torchvision、torchaudio、torch-cluster、torch-sparse 和 torch-geometric，当前源码没有直接导入这些包，因此最小依赖中省略。

从仓库根目录编译高度计算扩展：

```powershell
cd utils/cpp_wrappers/cpp_height
python setup.py install
python test.py
cd ../../..
```

Windows 需要 Visual Studio C++ Build Tools。现有 `setup.py` 保留原有 GCC 编译选项，MSVC 可能提示忽略这些选项；扩展编译和 GPU 兼容性需在目标环境验证。

## 数据准备

数据集须自行获取。UrbanBIS 预处理读取 TXT，每行至少包含 `x y z r g b class` 七列；RGB 使用 0–255，类别编号为 0–6。脚本按以下原始结构查找数据：

```text
<dataset_root>/train/<scene>/train/*.txt
<dataset_root>/train/<scene>/val/*.txt
<dataset_root>/test/<scene>/*.txt
```

始终从仓库根目录运行模块，避免 `utils` 导入路径错误：

```powershell
python -m utils.preprocess.preprocess_urbanbis --dataset_root "D:/datasets/UrbanBIS" --output_root "data/UrbanBIS" --pillar_size 0.5 --keep_points_num 3
```

默认场景为 Lihu、Longhua、Qingdao、Wuhu、Yingrenshi、Yuehai，可用 `--scene_name_list Lihu Longhua` 覆盖。输出位于 `data/UrbanBIS/train`、`val`、`test`，均不会被 Git 跟踪。

SensatUrban 预处理源码也予以保留，但当前未提供对应训练配置和专用数据加载器，不能仅替换数据路径就视为完整训练支持。

## 训练

确认 `configs/urban_seg_train.py` 中的 `data_root`、batch size、epoch 等参数符合本地数据和显存容量，然后在仓库根目录运行：

```powershell
python main.py --mode train --exp-name urban_seg
```

训练日志和检查点写入 `result/urban_seg/`，最佳权重为 `result/urban_seg/model/model_best.pth`。当前命令行从头训练，不提供断点续训选项；`trainer.py` 保留原有续训实现。

## 单场景推理

输入支持带 XYZ/RGB 的 PLY（字段 `x y z red green blue`），或前六列为 XYZ/RGB 的 TXT。需自行准备与模型配置一致的检查点，包含 `state_dict` 和 `epoch`：

```powershell
python main.py --mode test --input "data/scene.ply" --weight "result/urban_seg/model/model_best.pth" --exp-name urban_seg
```

可在 `--input` 后指定多个文件。默认使用 `configs/urban_seg.py`，`--config` 可选择 `configs/` 下其他配置模块（不含 `.py`）。输出写入 `result/<exp-name>/save/`，包含坐标、原始 RGB 及 `pred` 字段。TXT 输入也输出为 PLY 格式。

预处理包含随机采样，预测输出对应预处理后保留的点，并非输入文件的全部点。单场景推理不计算带真值的测试集指标。

| 标签 | 类别 |
| --- | --- |
| 0 | Terrain / 地面 |
| 1 | Vegetation / 植被 |
| 2 | Water / 水体 |
| 3 | Bridge / 桥梁 |
| 4 | Vehicle / 车辆 |
| 5 | Boat / 船舶 |
| 6 | Building / 建筑 |

可修改 `LabelsFileGenerate.py` 中的 `vis_path` 和 `output`，再运行该脚本生成语义 RGB 点云及标签 TXT。`plyRGBcorrect.py` 同样需要先设置自己的输入、输出路径。

## 源码发布说明

此版本保留全部 Python 与 C++ 源码，合并配置目录，新增参数化入口，修复预处理覆盖数据路径、配置模块路径分隔符及推理丢弃尾批次的问题。原始工作目录不受影响。

`.gitignore` 排除了数据、权重、预测、日志、缓存、编译产物及 IDE 文件。不要使用 `git add -f` 强行加入这些文件。上传前可用 `git status --short` 检查暂存内容。

## 许可证

本项目的原创贡献采用 [MIT License](LICENSE)，允许使用、修改、再分发及商业使用，须保留版权声明和许可文本。

第三方代码仍受其原有许可证约束，项目级 MIT 许可不替代上游许可。已保留源码中的作者及版权声明，包括 `z_order.py` 的 MIT 声明和 `nanoflann.hpp` 的 BSD 许可文本。部分文件包含 Point Transformer V3 及其他上游作者声明，其完整来源和许可尚未核实，发布前需补齐相应上游许可文本及所需声明。

本次整理仅进行了 Python 语法检查、命令行入口检查及发布文件核对，未进行 C++ 编译、GPU 训练或推理验证。
