import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from abc import ABC, abstractmethod


class Model(ABC):
    """Model base class."""

    @abstractmethod
    def __init__(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
    ) -> None:
        """Initialize parameters for base Model."""
        self._lower = lower
        self._upper = upper

    @property
    @abstractmethod
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        pass

    def evaluate_samples(
        self,
        samples: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate list of samples and return fitness in order."""
        raise NotImplementedError

    @property
    def lower(self):
        """The lower property."""
        return self._lower

    @lower.setter
    def lower(self, value):
        self._lower = value

    @property
    def upper(self):
        """The upper property."""
        return self._upper

    @upper.setter
    def upper(self, value):
        self._upper = value

    def search_range(self) -> npt.NDArray:
        """Make initial range for sample."""
        return np.array((self._lower, self._upper))

    def plot_fitness(
        self, ax: Axes, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Display the fitness.

        Only needed if plotting is desired.
        """
        raise NotImplementedError

    def plot_samples(self, ax: Axes, samples: npt.NDArray[np.float64]) -> None:
        """Display the samples.

        Only needed if plotting is desired.
        """
        raise NotImplementedError
