import numpy as np


class Optimizer:
    """Optimizer class orchestrating the Model and Distribution."""

    def optimize(self) -> None:
        """Differentiate optimization from return logic."""
        raise NotImplementedError

    def get_optimum(self) -> np.ndarray:
        """Get optimum sample."""
