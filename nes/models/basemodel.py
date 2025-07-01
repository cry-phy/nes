import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from abc import ABC, abstractmethod
from collections.abc import Iterable


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

    def __radd__(self, other):
        """Support scalar + model."""
        if isinstance(other, (int, float)):
            return ShiftedModel(self, other)
        return NotImplemented

    def __add__(self, other):
        """Support model + scalar or model + model."""
        if isinstance(other, Model):
            return AddComposite.from_iter([self, other])
        elif isinstance(other, (int, float)):
            return ShiftedModel(self, other)
        return NotImplemented

    def __sub__(self, other):
        """Return a model representing self - other."""
        if isinstance(other, Model):
            return SubComposite.from_iter([self, other])
        return NotImplemented

    def __rsub__(self, other):
        """Return a model representing other - self."""
        if isinstance(other, Model):
            return SubComposite.from_iter([other, self])
        return NotImplemented

    def __mul__(self, other):
        """Return a model representing self * other."""
        if isinstance(other, Model):
            return MultComposite.from_iter([self, other])
        return NotImplemented

    def __rmul__(self, other):
        """Return a model representing scalar * self."""
        if isinstance(other, (int, float)):
            return ScaledModel(self, other)
        return NotImplemented

    def __truediv__(self, other):
        """Return a model representing self / other."""
        if isinstance(other, Model):
            return DivComposite.from_iter([self, other])
        return NotImplemented

    def __rtruediv__(self, other):
        """Return a model representing other / self."""
        if isinstance(other, Model):
            return DivComposite.from_iter([other, self])
        return NotImplemented

    def __iadd__(self, other):
        """In-place addition of another model."""
        return self.__add__(other)

    def __isub__(self, other):
        """In-place subtraction of another model."""
        return self.__sub__(other)

    def __imul__(self, other):
        """In-place multiplication of another model."""
        return self.__mul__(other)

    def __itruediv__(self, other):
        """In-place division by another model."""
        return self.__truediv__(other)

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


class ScaledModel(Model):
    """Model representing scalar * base_model."""

    def __init__(self, base: Model, scale: float) -> None:
        lower = base.lower * scale
        upper = base.upper * scale
        super().__init__(lower, upper)
        self.base = base
        self.scale = scale

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate samples then scale the output."""
        super().evaluate_samples(samples)
        return self.scale * self.base.evaluate_samples(samples)


class ShiftedModel(Model):
    """Model representing base_model + constant scalar."""

    def __init__(self, base: Model, shift: float) -> None:
        lower = base.lower + shift
        upper = base.upper + shift
        super().__init__(lower, upper)
        self.base = base
        self.shift = shift

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate samples and add the constant shift."""
        super().evaluate_samples(samples)
        return self.base.evaluate_samples(samples) + self.shift


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
    def from_iter(cls, iterable: Iterable[Model]) -> "Composite":
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
        return result


class AddComposite(Composite):
    """Model consisting of different models of same degrees of freedom."""

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Combine returns from all models."""
        super().evaluate_samples(samples)
        result = np.zeros((samples.shape[0], 1))
        for m in self._models:
            result += m.evaluate_samples(samples)
        return result


class SubComposite(Composite):
    """Model consisting of different models of same degrees of freedom."""

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Combine returns from all models."""
        super().evaluate_samples(samples)
        result = np.zeros((samples.shape[0], 1))
        for m in self._models:
            result -= m.evaluate_samples(samples)
        return result


class MultComposite(Composite):
    """Model consisting of different models of same degrees of freedom."""

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Combine returns from all models."""
        super().evaluate_samples(samples)
        result = np.ones((samples.shape[0], 1))
        for m in self._models:
            result *= m.evaluate_samples(samples)
        return result


class DivComposite(Composite):
    """Model consisting of different models of same degrees of freedom."""

    @classmethod
    def from_iter(cls, iterable: Iterable[Model]) -> "Composite":
        """Generate a composite model from a list of model."""
        try:
            _, _ = iterable
        except ValueError:
            raise ValueError("Exactly 2 models required for division.") from None
        return super().from_iter(iterable)

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Combine returns from all models."""
        super().evaluate_samples(samples)
        result = np.ones((samples.shape[0], 1))
        for m in self._models:
            result *= m.evaluate_samples(samples)
        return result
