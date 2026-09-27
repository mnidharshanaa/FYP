import random

import numpy as np

from src.utils.seeding import set_global_seed


def test_python_random_reproducible():
    set_global_seed(42)
    a = [random.random() for _ in range(5)]
    set_global_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_reproducible():
    set_global_seed(42)
    a = np.random.rand(5).tolist()
    set_global_seed(42)
    b = np.random.rand(5).tolist()
    assert a == b


def test_different_seeds_diverge():
    set_global_seed(1)
    a = [random.random() for _ in range(5)]
    set_global_seed(2)
    b = [random.random() for _ in range(5)]
    assert a != b
