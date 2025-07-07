from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve


class Normal(Dist):
    """Multivariate Normal distribution.

    Limited to diagonal covariance for now as there exists a closed form of cdf.
    """

    var_eps = 1e-8
    alpha = 0.2

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        !!! Use factory methods instead !!!
        """
        super().__init__(num_dimensions)
        # A_var is the chowsky decomposition of covariance matriy A^T @ A = Covariance
        # Save only A maybe
        self._param_A_var: npt.NDArray[np.float64] = np.eye(num_dimensions)
        self._param_mean: npt.NDArray[np.float64] = np.zeros(num_dimensions)

    @property
    def _mean(self) -> npt.NDArray[np.float64]:
        """The mean value."""
        return self._param_mean.copy()

    @_mean.setter
    def _mean(self, mean) -> None:
        """The mean value."""
        self._param_mean[: self._num_dimensions] = mean

    @property
    def _variance(self) -> npt.NDArray[np.float64]:
        """The dimensional sigmas."""
        return self._param_A_var.T @ self._param_A_var

    def safe_variance(self) -> npt.NDArray[np.float64]:
        """Limit variance to positive values."""
        return np.maximum(self._variance, self.var_eps)

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "Normal":
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
        samples = self._mean.reshape((len(self._mean), 1)) + np.matmul(
            self._param_A_var, z.T
        )
        return samples.T

    def _gradient(
        self, samples: npt.NDArray, fitnesses: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Compute gradient from samples."""
        gradient_mean = np.zeros(shape=(self._num_dimensions))
        gradient_A_var = np.zeros(shape=(self._num_dimensions, self._num_dimensions))
        fischer_mu = np.zeros(shape=(self._num_dimensions, self._num_dimensions))
        fischer_A = np.zeros(shape=(self._num_dimensions, self._num_dimensions))
        inverse_A_var = solve_triangular(
            self._param_A_var, np.eye(self._num_dimensions)
        )
        invVariance = inverse_A_var @ inverse_A_var.T
        adjusted_fitness = -fitnesses + fitnesses.max()
        adjusted_fitness /= adjusted_fitness.sum()
        for idx, sample in enumerate(samples):
            log_grad_mean = invVariance @ (sample - self._mean)
            log_grad_variance = -0.5 * invVariance + 0.5 * np.outer(
                invVariance @ (sample - self._mean),
                (sample - self._mean).T @ invVariance,
            )
            gradient_mean += adjusted_fitness[idx] * log_grad_mean
            gradient_A_var += (
                adjusted_fitness[idx]
                * self._param_A_var
                @ (log_grad_variance + log_grad_variance.T)
            )
            fischer_mu += np.outer(log_grad_mean, log_grad_mean)
            fischer_A += log_grad_variance @ log_grad_variance.T
        return gradient_mean, gradient_A_var, fischer_mu, fischer_A

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray:
        """Get variation of parameters."""
        grad_mean, grad_A, fischer_mu, fischer_A = self._gradient(samples, fitnesses)
        eps = 1e-6
        fischer_mu += eps * np.eye(self._num_dimensions)
        fischer_A += eps * np.eye(self._num_dimensions)
        cho_fischer_mu, lower = cho_factor(fischer_mu)
        cho_fischer_A, lower = cho_factor(fischer_A)
        # Solving equation F@variation = beta grad_parameters for variation to avoid
        # Having to invert the Fisher matrix fully
        variation_mean = cho_solve((cho_fischer_mu, lower), grad_mean)
        variation_A = cho_solve((cho_fischer_A, lower), grad_A)
        self._param_mean += variation_mean
        self._param_A_var += variation_A
        # Handeling variances to return in a list so optimizer can handle return value
        variation = np.concatenate((variation_mean, np.diag(variation_A)))
        return variation

    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum parameters."""
        return self._mean.copy()

    def set_init_params(
        self, means: npt.NDArray[np.float64], variance: npt.NDArray[np.float64]
    ) -> None:
        """Get initial parameters."""
        self._param_mean = means
        # self._param_variance = variance
        self._param_A_var = cholesky(variance, lower=False)

    def plot_sigma(self, ax, num=3) -> None:
        """Draw ellipsis of sigma bound."""
        pass

    def get_covariance(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._param_A_var.T @ self._param_A_var

    def get_mean(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._mean.copy()


class NormalSchaulLR(Normal):
    """Use Constant learning rates heuristically found by Schaul et al."""

    def __init__(self, num_dimensions: int) -> None:
        super().__init__(num_dimensions)
        mu_rates = np.full(self._num_dimensions, 1)
        var_rates = np.full(
            self._num_dimensions,
            (3 + np.log(self._num_dimensions)) / 5 / np.sqrt(self._num_dimensions),
        )
        self.rates = np.concatenate((mu_rates, var_rates))

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray:
        """Get variation of parameters."""
        grad_mean, grad_A, fischer_mu, fischer_A = self._gradient(samples, fitnesses)
        cho_fischer_mu, lower = cho_factor(fischer_mu)
        cho_fischer_A, lower = cho_factor(fischer_A)
        # Solving equation F@variation = beta grad_parameters for variation to avoid
        # Having to invert the Fisher matrix fully
        variation_mean = cho_solve((cho_fischer_mu, lower), grad_mean)
        variation_A = cho_solve((cho_fischer_A, lower), grad_A)
        self._param_mean += self.rates[0] * variation_mean
        self._param_A_var += self.rates[1] * variation_A
        # Handeling variances to return in a list so optimizer can handle return value
        variation = np.concatenate((variation_mean, np.diag(variation_A)))
        return variation

    def get_covariance(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._param_A_var.T @ self._param_A_var.T

    def get_mean(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._mean.copy()
