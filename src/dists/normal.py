from dists.basedist import Dist
import numpy as np


class DiagNormal(Dist):
    """Multivariate Normal distribution.

    Limited to diagonal covariance for now as there exists a closed form of cdf."""

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        Use factory methods instead!
        """
        super().__init__(num_dimensions)
        self._params: np.float64 = np.zeros(2 * num_dimensions)

    @property
    def _mean(self) -> np.ndarray[np.float64]:
        """The mean value."""
        return self._params[: self._num_dimensions]

    @property
    def _variance(self) -> np.ndarray[np.float64]:
        """The dimensional sigmas."""
        return self._params[self._num_dimensions :]

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "DiagNormal":
        """Generate Normal distribution with number of dimensions."""
        if num_dimensions < 1:
            raise ValueError("Must be more than 0 dimensions.")
        return cls(num_dimensions)

    def get_samples(self, number: int) -> np.ndarray[np.float64]:
        """Generate samples."""
        z = np.random.randn(number, self._num_dimensions)
        return self._mean + z * np.sqrt(self._variance)

    def _gradient(self, samples: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Compute gradient from samples."""
        gradients = np.zeros(shape=(len(samples), self.num_params))
        diffs = gradients[:, self._num_dimensions :] - self._mean
        gradients[:, : self._num_dimensions] = diffs / self._variance
        gradients[:, self._num_dimensions :] = (
            0.5 * (diffs**2 / self._variance - 1) / self._variance
        )
        weighted_gradients = gradients.T * fitnesses / fitnesses.sum()
        return weighted_gradients.T.sum(axis=0)

    def _inverse_fisher(self, samples: np.ndarray) -> np.ndarray:
        """Compute Fisher matrix.

        Since there exists a closed form we simply use it here.
        """
        # save fisher as class variable so its not reallocated each time.
        fisher = self._params.copy()
        fisher[: self._num_dimensions] = fisher[: self._num_dimensions]
        fisher[self._num_dimensions :] = 2 * fisher[self._num_dimensions :] ** 2
        return fisher

    def param_variation(self, samples: np.ndarray, fitnesses: np.ndarray) -> None:
        """Get variation of parameters."""
        self._params += self._inverse_fisher(samples) * self._gradient(
            samples, fitnesses
        )
