import utils.augmentations.misc as misc


class AugmentationBuilder(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.augmentation = []
        for aug_cfg in self.cfg:
            aug_dict = aug_cfg.copy()
            aug_class = getattr(misc, aug_dict.pop('type'))(**aug_dict)
            self.augmentation.append(aug_class)

    def __call__(self, data_dict):
        for augmentation in self.augmentation:
            data_dict = augmentation(data_dict)
        return data_dict
