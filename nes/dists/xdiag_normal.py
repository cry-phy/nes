from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt


class XDiagNormal(Dist):
    """Logarithmic representation of covariance parameters.

    Parameter layout:
        [mu_0, mu_n, ln(sigma_0), ..., ln(sigma_n)]
    """

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        Use factory methods instead!
        """
        super().__init__(num_dimensions)
        self._params: npt.NDArray[np.float64] = np.zeros(2 * num_dimensions)

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
        invfisher[: self._num_dimensions] = invfisher[: self._num_dimensions]
        invfisher[self._num_dimensions :] = 1
        return invfisher
