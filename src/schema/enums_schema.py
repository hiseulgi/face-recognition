import rootutils

ROOT = rootutils.autosetup()

from enum import Enum


class DetectionModel(str, Enum):
    FACEONNX = "faceonnx"


class RecognitionModel(str, Enum):
    ARCFACE = "arcface"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
