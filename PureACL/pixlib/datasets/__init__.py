from PureACL.pixlib.utils.tools import get_class
from PureACL.pixlib.datasets.base_dataset import BaseDataset


def get_dataset(name):
    return get_class(name, __name__, BaseDataset)
