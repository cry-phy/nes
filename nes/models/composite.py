from typing import Iterable
from nes.models import Model
import numpy as np
import numpy.typing as npt


class Composite(Model):
    """Model consisting of different models of same degrees of freedom."""

    def __init__(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Instanciate variables at instance level."""
        super().__init__(lower, upper)
        self._dof = 0
        self._models = []

    @classmethod
    def from_iter(cls, iterable: Iterable[Model]) -> "Composite":
        """Generate a composite model from a list of model."""
        dof = next(iter(iterable)).dof
        upper = np.full(dof, np.inf)
        lower = np.full(dof, -np.inf)
        for model in iterable:
            if model.dof != dof:
                raise ValueError("All models need to have the same degrees of freedom!")
            upper = np.maximum(upper, model.upper)
            lower = np.minimum(lower, model.lower)
        instance = cls(lower, upper)
        instance._dof = dof
        instance._models = list[iterable]
        return instance

    def dof(self):
        """Return degrees of freedom."""
        return self._dof

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Combine returns from all models."""
        super().evaluate_samples(samples)
        result = np.zeros((samples.shape[0], 1))
        for model in self._models:
            result += model.evaluate_samples(samples)
        return result
