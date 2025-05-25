from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt
from  scipy.linalg import cholesky, solve_triangular


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
        self._param_variance: npt.NDArray[np.float64] = np.eye(num_dimensions)
        # A_var is the chowsky decomposition of covariance matriy A^T @ A = Covariance
        # Save only A maybe
        self._param_A_var: npt.NDArray[np.float64] = np.eye(num_dimensions)
        self._param_invA_var: npt.NDArray[np.float64] = np.eye(num_dimensions)
        self._param_mean: npt.NDArray[np.float64] = np.zeros(num_dimensions)


    @property
    def _mean(self) -> npt.NDArray[np.float64]:
        """The mean value."""
        return self._param_mean.copy()

    @property
    def _variance(self) -> npt.NDArray[np.float64]:
        """The dimensional sigmas."""
        return self._param_variance.copy()

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
        return self._mean + self._param_A_var.T @ z

    def _gradient(self, samples: npt.NDArray, fitnesses: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute gradient from samples."""
        gradient_mean = np.zeros(shape=(self._num_dimensions))
        gradient_variance = np.zeros(shape=(self._num_dimensions, self._num_dimensions))
        self._param_invA_var = solve_triangular(self._param_A_var, np.eye(self._num_dimensions))
        invVariance = self._param_invA_var @ self._param_invA_var.T
        for idx, sample in enumerate(samples):
            gradient_mean += fitnesses[idx] * invVariance @ (sample - self._mean)
            gradient_variance += fitnesses[idx] * (-0.5 * invVariance
                                      + 0.5 * (invVariance @ (sample-self._mean)
                                                @ (sample-self._mean).T @ invVariance
                                      )
            )
        gradient_mean /= len(samples)
        gradient_variance /= len(samples)
        return gradient_mean, gradient_variance



    def _inverse_fisher(self, samples: npt.NDArray) -> npt.NDArray:
        """Compute Fisher matrix.

        Since there exists a closed form we simply use it here.
        """
        pass


    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Get variation of parameters."""
        pass

    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum parameters."""
        return self._mean.copy()

    def set_init_params(self, means: npt.NDArray[np.float64], variance: npt.NDArray[np.float64]) -> None:
        """Get initial parameters."""
        self._param_mean = means
        self._param_variance = variance
        self._param_A_var = cholesky(variance, lower=False)

    def plot_sigma(self, ax, num=3) -> None:
        """Draw ellipsis of sigma bound."""
        pass
