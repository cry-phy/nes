import numpy as np
from matplotlib.axes import Axes


class Model:
    """Model base class."""

    def __init__(
        self,
        lower: np.ndarray[np.float64],
        upper: np.ndarray[np.float64],
    ) -> None:
        self._lower = lower
        self._upper = upper

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        raise NotImplementedError

    def evaluate_samples(
        self,
        samples: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
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

    def search_range(self) -> np.ndarray:
        """Make initial guess for sample."""
        raise NotImplementedError

    def plot_fitness(
        self, ax: Axes, lower: np.ndarray[np.float64], upper: np.ndarray[np.float64]
    ) -> None:
        """Display the fitness."""
        raise NotImplementedError

    def plot_samples(self, ax: Axes, samples: np.ndarray[np.float64]) -> None:
        """Display the samples."""
        raise NotImplementedError
