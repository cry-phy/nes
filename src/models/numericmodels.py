from models.basemodel import Model
import numpy as np


class NPolynome(Model):
    """d->1 dimensional Polynome.

    Contains:
    - Optimum Location
    - Directional scalings
    """

    def __init__(self, num_dims: int, weights: np.ndarray[np.float64]) -> None:
        """Initialize high dimensional parabola.

        Weights w up to max order n and dimension d are structured:
        [
            [w_00, w_01, ..., w_0d]
            [w_10, w_11, ..., w_1d]
            [ ...,  ..., ..., ...,]
            [w_n0, w_n0, ..., w_nd]
        ]
        """
        self.num_dims: int = num_dims
        self.weights: np.ndarray[np.float64] = weights

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        return self.num_dims

    def evaluate_samples(
        self, samples: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Evaluate list of samples and return fitness in order.

        Broadcasting is a bitch. Counter: 2.5h
        """
        orders = np.arange(self.weights.shape[0])
        weights_expanded = self.weights.T[np.newaxis, :, :]
        exponentiated = samples[:, :, np.newaxis] ** orders[np.newaxis, np.newaxis, :]
        return (weights_expanded * exponentiated).sum(axis=(1, 2))


class Spherical(Model):
    """Spherical Harmonics."""

    def __init__(self, l_order: int, m_order: int) -> None:
        """Initialize Spherical Harmonic of order l, m."""
        super().__init__()
        self.l: int = l_order
        self.m: int = m_order

    def __ass_legendre(self, value: float) -> float:
        """Evaluate associated Legendre Polynome."""
        ...

    def __gen_coefficients(self) -> None:
        """Generate Coefficients for the associated Legendre Polynome."""
        ...

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        return 2

    def evaluate_samples(
        self, samples: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Evaluate list of samples and return fitness in order."""
        raise NotImplementedError

    def initial_guess(self) -> np.ndarray:
        """Make initial guess for sample."""
        raise NotImplementedError
