from numpy.testing import assert_allclose
from nes.models import Composite, NPolynome
import numpy as np


def test_from_iter():
    """Test creation of Composite model."""
    num = 10
    models = [NPolynome(np.empty((0, 0))) for _ in range(num)]
    Composite.add_iter(models)


def test_composite_evaluate():
    """Test evaluation of Composite class."""
    num = 10
    dims = 10
    value = 1
    num_samples = 1
    models = [NPolynome(np.ones((value, dims))) for _ in range(num)]
    model = Composite.add_iter(models)
    samples = np.zeros((num_samples, 1))
    fitness = model.evaluate_samples(samples)
    expected = np.full((num_samples, 1), dims * value * num)
    assert_allclose(expected, fitness)


def test_model_add():
    """Test model __add__ method."""
    model1 = NPolynome(np.empty((0, 0)))
    model2 = NPolynome(np.empty((0, 0)))
    _ = model1 + model2
