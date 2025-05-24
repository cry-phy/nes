from nes.dists import Dist
from nes.dists.utils import grad_dep_rate
from nes.models import Model
import numpy as np
import numpy.typing as npt


class XDiagNormal(Dist):
    """Logarithmic representation of covariance parameters.

    Parameter layout:
        [mu_0, mu_n, ln(sigma_0), ..., ln(sigma_n)]
    """

    var_eps = 1e-8
    alpha = 0.2

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        Use factory methods instead!
        """
        super().__init__(num_dimensions)
        self._params: npt.NDArray[np.float64] = np.zeros(2 * num_dimensions)
        self._var_ma: float = 1e6
        self._fac_ma: float = 0.3

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "Dist":
        """Generate empty Ditsribution."""
        return cls(num_dimensions)

    @classmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model directly."""
        lower, upper = model.search_range()
        instance = cls(len(lower))
        means = (upper + lower) / 2
        variances = 2 * np.log(upper - lower) + np.log(6)
        params = np.concatenate((means.reshape(-1), variances.reshape(-1)))
        instance.set_init_params(params)
        return instance

    @property
    def _mean(self) -> npt.NDArray[np.float64]:
        """The mean value."""
        return self._params[: self._num_dimensions].copy()

    @property
    def _variance(self) -> npt.NDArray[np.float64]:
        """The dimensional sigmas."""
        return np.exp(2 * self._params[self._num_dimensions :])

    def get_samples(self, number: int) -> npt.NDArray[np.float64]:
        """Generate samples."""
        z = np.random.randn(number, self._num_dimensions)
        return self._mean + z * np.sqrt(self._variance)

    def _gradient(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute gradient from samples."""
        gradients = np.zeros(shape=(len(samples), self.num_params))
        diffs = samples - self._mean
        gradients[:, : self._num_dimensions] = diffs / self._variance
        gradients[:, self._num_dimensions :] = diffs**2 / self._variance - 1
        adjusted_fitness = fitnesses - fitnesses.min()
        weighted_gradients = gradients * (adjusted_fitness / adjusted_fitness.sum())
        return weighted_gradients.sum(axis=0)

    def _inverse_fisher(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Analytically compute Fisher matrix."""
        invfisher = self._params.copy()
        invfisher[: self._num_dimensions] = self._variance
        invfisher[self._num_dimensions :] = 1
        return invfisher

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute standard Parameter variation."""
        variation = -self._inverse_fisher(samples) * self._gradient(samples, fitnesses)
        self._params += self._adaptive_rate(variation) * variation
        diff_ma = np.linalg.norm(variation[self._num_dimensions :]) - self._var_ma
        self._var_ma = float((1 - self._fac_ma) * self._var_ma + self._fac_ma * diff_ma)
        return variation

    def _adaptive_rate(self, variation) -> npt.NDArray:
        eta = np.concatenate(
            [
                grad_dep_rate(variation[: self._num_dimensions]),
                grad_dep_rate(variation[self._num_dimensions :])
                * self._stagnation_factor(),
            ]
        )
        return eta

    def _stagnation_factor(
        self,
        lower: float = 0,
        upper: float = 20,
        steep: float = 1,
        thresh: float = 1e-2,
    ) -> float | npt.NDArray:
        """Adjust variance according to stagnation of parameters."""
        return lower + (upper - lower) * 1 / (
            1 + np.exp(steep * (self._var_ma - thresh))
        )

    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum mean parameters."""
        return self._mean

    def set_init_params(self, parameters: npt.NDArray[np.float64]) -> None:
        """Get initial parameters."""
        if np.any(parameters[self._num_dimensions :] <= 0):
            raise ValueError("Cannot use negative variances.")
        self._params[: self._num_dimensions] = parameters[: self._num_dimensions]
        self._params[self._num_dimensions :] = np.log(
            parameters[self._num_dimensions :]
        )
