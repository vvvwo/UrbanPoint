import importlib
from os.path import join, exists, dirname


def get_module(name, type):
    try:
        return importlib.import_module(f"utils.{type}.{name}")
    except:
        raise OSError("The specified module does not exist in utils")
