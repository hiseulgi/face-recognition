"""Math utility functions."""

import rootutils

ROOT = rootutils.autosetup()

import numpy as np


def find_cosine_distance(src: np.array, dst: np.array) -> float:
    """
    Find cosine distance between two face representations.

    Args:
        src (np.array): Source face representation.
        dst (np.array): Destination face representation.

    Returns:
        float: Cosine distance.

    Examples:
        >>> find_cosine_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))
        0.9746318461970762
    """
    # convert to numpy array
    if isinstance(src, list):
        src = np.array(src)
    if isinstance(dst, list):
        dst = np.array(dst)
    if isinstance(src, tuple):
        src = list(src)
        src = np.array(src)
    if isinstance(dst, tuple):
        dst = list(dst)
        dst = np.array(dst)

    a = np.matmul(np.transpose(src), dst)
    b = np.sum(np.multiply(src, src))
    c = np.sum(np.multiply(dst, dst))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
