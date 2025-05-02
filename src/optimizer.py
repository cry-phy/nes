from typing import Callable
from dists.basedist import Dist
from models.basemodel import Model
import numpy as np


class Optimizer:
    """Optimizer class orchestrating the Model and Distribution."""

    def __init__(self, model: Model, dist: Dist, sample_num: int) -> None:
        """Initialize Optimizer.

        Set with preset Model and distribution.
        """
        self.model: Model = model
        self.dist: Dist = dist

        self.sample_num = sample_num

        self._tolerance: float = 1e-3
        self.decider: Callable[[], bool] = self.gradient_decider

        self._gradient: np.ndarray[np.float64] = np.array([np.inf])

    @property
    def tolerance(self) -> float:
        """The tolerance property."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value) -> None:
        self._tolerance = value

    def gradient_decider(self) -> bool:
        """Decide continuation based on tolerance in gradient."""
        return self._gradient.sum() > self.tolerance

    def optimize(self) -> None:
        """Differentiate optimization from return logic."""
        while self.decider():
            samples = self.dist.get_samples(self.sample_num)
            fitness = self.model.evaluate_samples(samples)
            self.dist.param_variation(samples, fitness)
        opt_loc = self.get_optimum()
        opt_fit = self.model.evaluate_samples([opt_loc])[0]
        print(f"Found optimum {opt_loc} with fitness {opt_fit}.")

    def get_optimum(self) -> np.ndarray:
        """Get optimum sample."""
        if self.decider():
            return self.dist.get_optimum()
