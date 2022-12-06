from pixloc.pixlib.utils.tools import get_class
from pixloc.pixlib.datasets.base_dataset import BaseDataset


def get_dataset(name):
    return get_class(name, __name__, BaseDataset)
