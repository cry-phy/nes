import numpy as np


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
