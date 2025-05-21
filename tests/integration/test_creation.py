from nes.dists import DiagNormal
from nes.models import NPolynome
import numpy as np

from nes import Optimizer


def test_init_creation():
    """Test if creation is working."""
    np.random.seed(0)
    dims = 10
    gradients = np.random.random(dims) * 2 - 1
    y_offsets = np.random.random(dims) * 2 - 1
    params = np.array([y_offsets, gradients])
    model = NPolynome(params)
    dist = DiagNormal(dims)
    _ = Optimizer(model, dist, 100)


def test_from_model():
    """Test generation pipeline from existing model."""
    np.random.seed(0)
    dims = 10
    gradients = np.random.random(dims) * 2 - 1
    y_offsets = np.random.random(dims) * 2 - 1
    params = np.array([y_offsets, gradients])
    model = NPolynome(params)
    model.upper = np.full((dims), 10)
    model.lower = np.full((dims), -10)
    dist = DiagNormal.from_model(model)
    _ = Optimizer(model, dist, 100)


def test_optimize_parabola():
    pass
