import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from abc import ABC, abstractmethod
from typing import Iterable


class Model(ABC):
    """Model base class."""

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

    @abstractmethod
    def evaluate_samples(
        self,
        samples: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate list of samples and return fitness in order."""
        pass

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

    def __add__(self, other: "Model") -> "Composite":
        """Generate a composite model from two models."""
        return Composite.add_iter([self, other])


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
    def add_iter(cls, iterable: Iterable[Model]) -> "Composite":
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
        instance._models = list(iterable)
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
