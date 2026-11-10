# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 0
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings
model = dict(
    name='UrbanSeg',
    num_classes=7,
    backbone_out_channels=64,
    in_channels=7,
    order=("z", "z-trans", "hilbert", "hilbert-trans", "u", "u-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(128, 128, 128, 128, 128),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(128, 128, 128, 128),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    pre_norm=True,
    adp_att=True,
)

# scheduler settings
epoch = 1
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]
loss = [
    dict(name="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    dict(name="CrossEntropyLoss",
         loss_weight=1.0, ignore_index=-1),
]

# dataset settings
dataset_type = "SingleUnlabeledDataset"

data = dict(
    num_classes=7,
    ignore_index=-1,
    names=['Terrain', 'Vegetation', 'Water', 'Bridge', 'Vehicle', 'Boat', 'Building'],
    test=dict(
        type=dataset_type,
        transform=[
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "color": "origin_color"},
            ),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.3,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
                keys=("coord", "color", "height")
            ),
            dict(type="CenterShift", apply_z=False),
            # dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_coord",
                    "origin_color",
                    "inverse",
                    "name"
                ),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=["coord", "color", "height"],
            ),
        ],
    ),
)
