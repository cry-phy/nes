from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt


class DiagNormal(Dist):
    """Multivariate Normal distribution.

    Limited to diagonal covariance for now as there exists a closed form of cdf.
    """

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        Use factory methods instead!
        """
        super().__init__(num_dimensions)
        self._params: npt.NDArray[np.float64] = np.zeros(2 * num_dimensions)

    @property
    def _mean(self) -> npt.NDArray[np.float64]:
        """The mean value."""
        return self._params[: self._num_dimensions]

    @property
    def _variance(self) -> npt.NDArray[np.float64]:
        """The dimensional sigmas."""
        return self._params[self._num_dimensions :]

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "DiagNormal":
        """Generate Normal distribution with number of dimensions."""
        if num_dimensions < 1:
            raise ValueError("Must be more than 0 dimensions.")
        return cls(num_dimensions)

    @classmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model directly."""
        lower, upper = model.search_range()
        instance = cls(len(lower))
        means = upper - lower / 2
        variances = np.sqrt(upper - lower)
        params = np.concatenate(means, variances)
        instance.set_init_params(params)
        return instance

    def get_samples(self, number: int) -> npt.NDArray[np.float64]:
        """Generate samples."""
        z = np.random.randn(number, self._num_dimensions)
        return self._mean + z * np.sqrt(self._variance)

    def _gradient(self, samples: npt.NDArray, fitnesses: npt.NDArray) -> npt.NDArray:
        """Compute gradient from samples."""
        gradients = np.zeros(shape=(len(samples), self.num_params))
        diffs = samples - self._mean
        gradients[:, : self._num_dimensions] = diffs / self._variance
        gradients[:, self._num_dimensions :] = (
            0.5 * (diffs**2 / self._variance - 1) / self._variance
        )
        weighted_gradients = gradients.T * fitnesses / fitnesses.sum()
        return weighted_gradients.T.sum(axis=0)

    def _inverse_fisher(self, samples: npt.NDArray) -> npt.NDArray:
        """Compute Fisher matrix.

        Since there exists a closed form we simply use it here.
        """
        # save fisher as class variable so its not reallocated each time.
        invfisher = self._params.copy()
        invfisher[: self._num_dimensions] = invfisher[: self._num_dimensions]
        invfisher[self._num_dimensions :] = 2 * invfisher[self._num_dimensions :] ** 2
        return invfisher

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> None:
        """Get variation of parameters."""
        self._params += (
            self.learning_rate
            * self._inverse_fisher(samples)
            * self._gradient(samples, fitnesses)
        )

    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum parameters."""
        return self._mean.copy()

    def set_init_params(self, parameters: npt.NDArray[np.float64]) -> None:
        """Get initial parameters."""
        self._params = parameters
