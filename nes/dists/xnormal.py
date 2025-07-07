from typing import cast
from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky, expm


class XNormal(Dist):
    """Multivariate Normal distribution.

    Limited to diagonal covariance for now as there exists a closed form of cdf.
    """

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        !!! Use factory methods instead !!!
        """
        super().__init__(num_dimensions)
        # A_var is the chowsky decomposition of covariance matriy A^T @ A = Covariance
        # Save only A maybe
        self._param_A_var: npt.NDArray[np.float64] = np.eye(num_dimensions)
        self._param_A_inv: npt.NDArray[np.float64] = np.eye(num_dimensions)
        self._param_mean: npt.NDArray[np.float64] = np.zeros(num_dimensions)

        self._eta_A = 0.5

    @property
    def _mean(self) -> npt.NDArray[np.float64]:
        """The mean property."""
        return self._param_mean

    @_mean.setter
    def _mean(self, mean) -> None:
        """The mean value."""
        self._param_mean[: self._num_dimensions] = mean

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "XNormal":
        """Generate Normal distribution with number of dimensions."""
        return cls(num_dimensions)

    @classmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model directly."""
        lower, upper = model.search_range()
        instance = cls(len(lower))
        means = (upper + lower) / 2
        variances = np.diag(((upper - lower) / 6) ** 2 / 10)
        instance.set_init_params(means, variances)
        return instance

    def get_samples(self, number: int) -> npt.NDArray[np.float64]:
        """Generate samples."""
        z = np.random.randn(number, self._num_dimensions)
        return self._mean + z @ self._param_A_var.T

    def _gradient(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate gradient."""
        diffs = (samples - self._mean) @ self._param_A_inv
        adjusted_fitness = -fitnesses + fitnesses.max()
        weights = adjusted_fitness / adjusted_fitness.sum()
        # can pull out I and sum w@I = 1 since sum w = 1
        gradient_matrix: npt.NDArray[np.float64] = np.einsum(
            "i,ij,ik->jk", weights.flatten(), diffs, diffs
        ) - np.eye(self._num_dimensions)
        gradient_matrix = 0.5 * (gradient_matrix + gradient_matrix.T)
        grad_mean = self._param_A_var @ np.sum(weights * diffs, axis=0)
        return grad_mean, cast(
            npt.NDArray[np.float64], expm(self._eta_A * gradient_matrix / 2)
        )

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Parameter variation."""
        grad_mean, grad_a = self._gradient(samples, fitnesses)
        self._param_mean += grad_mean
        self._param_A_var = self._param_A_var @ grad_a
        self._param_A_inv = cast(
            npt.NDArray[np.float64], np.linalg.inv(self._param_A_var)
        )
        return grad_mean, grad_a

    def set_init_params(
        self,
        mean: npt.NDArray[np.float64],
        covariance: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get initial parameters."""
        self._param_mean = mean.copy()
        self._param_A_var = cholesky(covariance)
        return mean, self._param_A_var.copy()

    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum parameters."""
        return self._mean.copy()

    def get_covariance(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._param_A_var.T @ self._param_A_var

    def get_mean(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._mean.copy()


class XNormalSchaulLR(XNormal):
    """Use Constant learning rates heuristically found by Schaul et al."""

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        !!! Use factory methods instead !!!
        """
        super().__init__(num_dimensions)
        self._eta_A = (3 + np.log(self._num_dimensions)) / (
            5 * np.sqrt(self._num_dimensions)
        )


class XNormalAdaptive(XNormal):
    """Use KL-based adaptive learning rates with EMA stabilization."""

    klbudget: float = 0.1
    ema_beta: float = 0.9

    def __init__(self, num_dimensions: int) -> None:
        super().__init__(num_dimensions)
        self._grad_norm_ema = 1.0  # For gradient matrix norm
        self._eta_mu = 1.0  # Fixed mean learning rate

    def _gradient(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Compute gradient matrix gradient_matrix explicitly, for adaptive KL scaling."""
        diffs = (samples - self._mean) @ self._param_A_inv
        adjusted_fitness = -fitnesses + fitnesses.max()
        weights = adjusted_fitness / adjusted_fitness.sum()
        weights = weights.flatten()

        gradient_matrix = np.einsum("i,ij,ik->jk", weights, diffs, diffs) - np.eye(
            self._num_dimensions
        )
        gradient_matrix = 0.5 * (gradient_matrix + gradient_matrix.T)
        grad_mu = self._param_A_var @ np.sum(weights[:, None] * diffs, axis=0)
        return grad_mu, gradient_matrix, weights

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute parameter update with KL-adaptive learning rate."""
        grad_mu, gradient_matrix, _ = self._gradient(samples, fitnesses)
        norm_sq = np.sum(gradient_matrix**2)
        self._grad_norm_ema = (
            self.ema_beta * self._grad_norm_ema + (1 - self.ema_beta) * norm_sq
        )
        self._eta_A = np.sqrt(self.klbudget / (self._grad_norm_ema + 1e-12))
        grad_a = cast(npt.NDArray[np.float64], expm(self._eta_A * gradient_matrix / 2))
        self._param_mean += self._eta_mu * grad_mu
        self._param_A_var = self._param_A_var @ grad_a
        self._param_A_inv = cast(
            npt.NDArray[np.float64], np.linalg.inv(self._param_A_var)
        )
        return grad_mu, grad_a
