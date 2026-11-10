import importlib
from .misc import *
from .lovasz import LovaszLoss


class LossBuilder:
    def __init__(self, cfg):
        self.cfg = cfg if cfg is not None else []
        self.losses = []
        for loss_cfg in self.cfg:
            name = loss_cfg.pop('name')
            if name == 'CrossEntropyLoss':
                loss_class = CrossEntropyLoss(**loss_cfg)
            elif name == 'SmoothCELoss':
                loss_class = SmoothCELoss(**loss_cfg)
            elif name == 'BinaryFocalLoss':
                loss_class = BinaryFocalLoss(**loss_cfg)
            elif name == 'FocalLoss':
                loss_class = FocalLoss(**loss_cfg)
            elif name == 'DiceLoss':
                loss_class = DiceLoss(**loss_cfg)
            elif name == 'LovaszLoss':
                loss_class = LovaszLoss(**loss_cfg)
            else:
                raise OSError("This Loss Function isn't supported")
            self.losses.append(loss_class)

    def __call__(self, pred, target):
        loss = 0
        for c in self.losses:
            loss += c(pred, target)
        return loss
