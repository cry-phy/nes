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

    def __add__(self, other: "Model") -> "Composite":
        """Generate a composite model from two models."""
        return Composite.add_iter([self, other])

    def plot_fitness(
        self,
        ax: Axes,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
        i1: int = 0,
        i2: int = 1,
    ) -> None:
        """Display the fitness against the first two dimensions."""
        n: int = 101
        line1 = np.linspace(lower[i1], upper[i1], n)
        line2 = np.linspace(lower[i2], upper[i2], n)
        x, y = np.meshgrid(line1, line2)
        samples_xy = np.array([x, y]).reshape(2, -1).T
        samples = np.zeros((n**2, self.dof))
        samples[:, i1] = samples_xy[:, 0]
        samples[:, i2] = samples_xy[:, 1]
        fitness = self.evaluate_samples(samples).reshape(x.shape)
        ax.contourf(x, y, fitness, levels=50, cmap="plasma")

    def plot_samples(
        self,
        ax: Axes,
        samples: npt.NDArray[np.float64],
        i1: int = 0,
        i2: int = 1,
    ) -> None:
        """Display the first two dimensions of samples."""
        xs: npt.NDArray = samples[:, i1]
        ys: npt.NDArray = samples[:, i2]
        ax.scatter(xs, ys, s=40, facecolors="none", edgecolors="b")


class Composite(Model):
    """Model consisting of different models of same degrees of freedom."""

    def __init__(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Instanciate variables at instance level."""
        super().__init__(lower, upper)
        self._dof: int = 0
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
            upper = np.minimum(upper, model.upper)
            lower = np.maximum(lower, model.lower)
        instance = cls(lower, upper)
        instance._dof = int(dof)
        instance._models = list(iterable)
        return instance

    @property
    def dof(self) -> int:
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
