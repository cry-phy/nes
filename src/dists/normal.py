from basedist import Dist
from dists.laplace import Laplace
import numpy as np


class Normal(Dist):
    """Multivariate Normal distribution."""

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        Use factory methods instead!
        """
        super().__init__(num_dimensions)
        self._envelope: Dist = Laplace.from_dimensions(num_dimensions)
        self._params: np.float64 = np.zeros(2 * num_dimensions)

    @property
    def _mean(self) -> np.ndarray[np.float64]:
        """The mean value."""
        return self._params[: self._num_dimensions]

    @property
    def _sigma(self) -> np.ndarray[np.float64]:
        """The dimesnional sigmas."""
        return self._params[self._num_dimensions :]

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "Normal":
        """Generate Normal distribution with number of dimensions."""
        return cls(num_dimensions)
