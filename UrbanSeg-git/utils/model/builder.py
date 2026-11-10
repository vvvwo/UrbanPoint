import torch.nn as nn
from .urban_seg import UrbanSeg
from utils.reflection import get_module
from utils import losses


class ModelBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        name = cfg.pop('name')
        if name == "UrbanSeg":
            self.network = UrbanSeg(**cfg)
        else:
            raise OSError("The selected model does not exist")

    def __call__(self, input_dict):
        if "condition" in input_dict.keys():
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.network(input_dict)
        return dict(seg_logits=seg_logits)
