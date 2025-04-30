import numpy as np


class Model:
    """Model base class."""

    @property
    def degrees_of_freedom(self):
        """The degrees_of_freedom property."""
        raise NotImplementedError

    def get_evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate list of samples and return fitness in order."""
        raise NotImplementedError
