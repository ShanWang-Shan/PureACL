from sidfm.pixlib.utils.tools import get_class
from sidfm.pixlib.models.base_model import BaseModel


def get_model(name):
    return get_class(name, __name__, BaseModel)
