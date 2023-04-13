from itertools import product
import numpy as np
from numpy.random import default_rng
import logging
from typing import Tuple, TypeVar

from ..data.data_class import TrainDataSet, TestDataSet

np.random.seed(42)
logger = logging.getLogger()


def y_x(x):
    return 0.8*x + 0.1*x**2 + 0.05*x**3


def generate_train_mr_design(n) -> TrainDataSet:
    U = np.random.normal(size=n)
    Z = np.random.normal(size=n)
    X = 0.75*Z + 0.15*U + np.random.normal(scale=0.5, size=n)

    struct = y_x(X)
    Y = struct + 0.2*U**2 + -U + np.random.normal(scale=0.5, size=n)

    return TrainDataSet(
        treatment=X.reshape(-1, 1),
        instrumental=Z.reshape(-1, 1),
        covariate=None,
        outcome=Y.reshape(-1, 1),
        structural=struct.reshape(-1, 1)
    )


def generate_test_mr_design() -> TestDataSet:
    x = np.linspace(-2.5, 2.5, 2000)
    test_data = TestDataSet(
        treatment=x.reshape(-1, 1),
        covariate=None,
        structural=y_x(x).reshape(-1, 1)
    )
    return test_data
