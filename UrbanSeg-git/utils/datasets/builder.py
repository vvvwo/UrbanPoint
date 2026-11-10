from .urbanbis import UrbanBISDataset
from .single import SingleUnlabeledDataset


def get_dataset(cfg, batch_size, path_list=None):
    if cfg["type"] == "UrbanBISDataset":
        return UrbanBISDataset(cfg, batch_size)
    if cfg["type"] == "SingleUnlabeledDataset":
        return SingleUnlabeledDataset(cfg, path_list)
    else:
        raise OSError("The selected dataset does not exist")
