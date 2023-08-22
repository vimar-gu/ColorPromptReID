from .resnet import *


__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError('Unknown model:', name)
    return __factory[name](*args, **kwargs)
