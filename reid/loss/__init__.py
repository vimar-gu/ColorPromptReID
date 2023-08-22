from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy, WeightedCrossEntropyLabelSmooth

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'WeightedCrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
]
