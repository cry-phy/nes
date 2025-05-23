from nes.models import NPolynome
import numpy as np
from numpy.testing import assert_allclose


def test_dof():
    """Test number of dims of NPolynome."""
    np.random.seed(0)
    param_dims = np.random.randint(0, 10, size=(10, 2))
    for n, d in param_dims:
        params = np.zeros((n, d))
        dims = NPolynome(params).dof
        assert dims == d


def test_1d_linear():
    """Ensure 1 dimensional linear polynome works as expected."""
    gradient = 3
    y_offset = -1
    dims = 1
    num_samples = 100
    params = np.array([[y_offset], [gradient]])
    model = NPolynome(params)
    samples = np.random.random((num_samples, dims))
    values = model.evaluate_samples(samples)
    check_values = gradient * samples + y_offset
    assert_allclose(check_values, values)


def test_nd_linear():
    """Ensure n dimensional linear polynome works as expected."""
    np.random.seed(0)
    dims = 10
    num_samples = 100
    gradients = np.random.random(dims) * 2 - 1
    y_offsets = np.random.random(dims) * 2 - 1
    params = np.array([y_offsets, gradients])
    model = NPolynome(params)
    samples = np.random.random((num_samples, dims))
    values = model.evaluate_samples(samples)
    check_values = np.array(
        [sum(gradients * sample + y_offsets) for sample in samples]
    ).reshape(-1, 1)
    assert_allclose(check_values, values)


def test_1d_cubic():
    """Test cubic function NPolynome."""
    np.random.seed(0)
    dims = 10
    num_samples = 100
    cubicians = np.random.random(dims) * 2 - 1
    parabolas = np.random.random(dims) * 2 - 1
    gradients = np.random.random(dims) * 2 - 1
    y_offsets = np.random.random(dims) * 2 - 1
    params = np.array([y_offsets, gradients, parabolas, cubicians])
    model = NPolynome(params)
    samples = np.random.random((num_samples, dims))
    values = model.evaluate_samples(samples)
    check_values = np.array(
        [
            sum(
                cubicians * sample**3
                + parabolas * sample**2
                + gradients * sample
                + y_offsets
            )
            for sample in samples
        ]
    ).reshape(-1, 1)
    assert_allclose(check_values, values)
