import numpy as np
from matplotlib.axes import Axes


class Model:
    """Model base class."""

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        raise NotImplementedError

    def evaluate_samples(
        self, samples: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Evaluate list of samples and return fitness in order."""
        raise NotImplementedError

    def initial_guess(self) -> np.ndarray:
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
