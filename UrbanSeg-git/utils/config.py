import importlib
from os.path import exists, splitext, join


class Config:
    def __init__(self, config_dic):
        for key, value in config_dic.items():
            setattr(self, key, value)


def get_config(config_name, options):
    path = join('configs', config_name + '.py')
    if not exists(path):
        raise RuntimeError("configs file is not exists")

    cfg_module = importlib.import_module(splitext(path)[0].replace("\\", '.').replace("/", "."))
    cfg_dict = {name: value for name, value in cfg_module.__dict__.items() if not name.startswith("__")}

    for key, value in options.items():
        cfg_dict[key] = value

    return Config(cfg_dict)
