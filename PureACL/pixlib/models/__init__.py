from PureACL.pixlib.utils.tools import get_class
from PureACL.pixlib.models.base_model import BaseModel


def get_model(name):
    return get_class(name, __name__, BaseModel)
