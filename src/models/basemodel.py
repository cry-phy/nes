import numpy as np


class Model:
    """Model base class."""

    def get_evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate list of samples and return fitness in order."""
        raise NotImplementedError
