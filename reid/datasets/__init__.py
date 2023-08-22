from .market1501 import Market1501
from .msmt17 import MSMT17
from .personx import PersonX
from .cuhksysu import CUHKSYSU
from .dukemtmcreid import DukeMTMCreID
from .cuhk02 import CUHK02
from .cuhk03 import CUHK03
from .prid import PRID
from .grid import GRID
from .ilids import iLIDS


__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'personx': PersonX,
    'cuhksysu': CUHKSYSU,
    'dukemtmc': DukeMTMCreID,
    'cuhk02': CUHK02,
    'cuhk03': CUHK03,
    'prid': PRID,
    'grid': GRID,
    'ilids': iLIDS,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError('Unknown dataset:', name)
    return __factory[name](root, *args, **kwargs)
