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

    def __rsub__(self, other):
        """Support scalar - model or model - model (reversed)."""
        if isinstance(other, (int, float)):
            return ShiftedModel(self, other)
        elif isinstance(other, Model):
            return SubComposite.from_iter([other, self])
        return NotImplemented

    def __sub__(self, other):
        """Support model - scalar or model - model."""
        if isinstance(other, Model):
            return SubComposite.from_iter([self, other])
        elif isinstance(other, (int, float)):
            return ShiftedModel(self, -other)
        return NotImplemented

    def __rmul__(self, other):
        """Support scalar * model."""
        if isinstance(other, (int, float)):
            return ScaledModel(self, other)
        return NotImplemented

    def __mul__(self, other):
        """Support model * scalar or model * model."""
        if isinstance(other, Model):
            return MultComposite.from_iter([self, other])
        elif isinstance(other, (int, float)):
            return ScaledModel(self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        """Support scalar / model or model / model (reversed)."""
        if isinstance(other, (int, float)):
            return DivComposite.from_iter([ConstantModel(self, other), self])
        elif isinstance(other, Model):
            return DivComposite.from_iter([other, self])
        return NotImplemented

    def __truediv__(self, other):
        """Support model / scalar or model / model."""
        if isinstance(other, Model):
            return DivComposite.from_iter([self, other])
        elif isinstance(other, (int, float)):
            return ScaledModel(self, 1.0 / other)
        return NotImplemented

    def __rpow__(self, base):
        """Support scalar ** model (i.e. exp(model * log(base)))."""
        if isinstance(base, (int, float)):
            return ExpModel(base, self)
        return NotImplemented

    def __pow__(self, power, modulo=None):
        """Support model ** scalar."""
        if isinstance(power, (int, float)):
            return ExponentModel(self, power)
        return NotImplemented

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __neg__(self):
        """Support -model (unary negation)."""
        return ScaledModel(self, -1.0)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy ufuncs like np.exp(model)."""
        if method != "__call__" or len(inputs) != 1 or inputs[0] is not self:
            return NotImplemented

        match ufunc:
            case np.exp:
                return self.exp()
            case np.sin:
                return self.sin()
            case np.cos:
                return self.cos()
            case np.log:
                return self.log()
            case np.abs:
                return self.abs()
            case np.sqrt:
                return self.sqrt()
            case _:
                return NotImplemented

    def abs(self) -> "AbsModel":
        """Return np.abs(self output)."""
        return AbsModel(self)

    def sqrt(self) -> "SqrtModel":
        """Return np.sqrt(self output)."""
        return SqrtModel(self)

    def exp(self) -> "ExpNPModel":
        """Return np.exp(self output)."""
        return ExpNPModel(self)

    def log(self) -> "LogModel":
        """Return np.log(self output)."""
        return LogModel(self)

    def sin(self) -> "SinModel":
        """Return np.sin()."""
        return SinModel(self)

    def cos(self) -> "CosModel":
        """Return np.cos()."""
        return CosModel(self)

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


class ConstantModel(Model):
    """TODO: properly set up this model from NPolynome."""

    def __init__(self, base: Model, constant: float) -> None:
        lower = base.lower
        upper = base.upper
        super().__init__(lower, upper)
        self.base = base
        self.constant = constant

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate samples then scale the output."""
        super().evaluate_samples(samples)
        array = self.base.evaluate_samples(samples)
        return np.full_like(array, self.constant)


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


class ExponentModel(Model):
    """Model representing (base_model output) ** exponent."""

    def __init__(self, base_model: Model, exponent: float) -> None:
        lower = np.power(base_model.lower, exponent)
        upper = np.power(base_model.upper, exponent)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        super().__init__(lower, upper)
        self.base_model = base_model
        self.exponent = exponent

    @property
    def dof(self) -> int:
        return self.base_model.dof

    def evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate samples and apply exponentiation."""
        super().evaluate_samples(samples)
        return np.power(self.base_model.evaluate_samples(samples), self.exponent)


class ExpModel(Model):
    """Model representing base**(base_model output)."""

    def __init__(self, base: float, base_model: Model) -> None:
        self.base = base
        lower = np.power(base, base_model.lower)
        upper = np.power(base, base_model.upper)
        super().__init__(lower, upper)
        self.base_model = base_model

    @property
    def dof(self) -> int:
        return self.base_model.dof

    def evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate model and apply base**output."""
        super().evaluate_samples(samples)
        return np.power(self.base, self.base_model.evaluate_samples(samples))


class LogModel(Model):
    """Model representing log(base_model output)."""

    def __init__(self, base: Model) -> None:
        if np.any(base.lower <= 0):
            raise ValueError("LogModel requires base model output > 0.")
        lower = np.log(base.lower)
        upper = np.log(base.upper)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        super().__init__(lower, upper)
        self.base = base

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate model and apply log."""
        super().evaluate_samples(samples)
        values = self.base.evaluate_samples(samples)
        if np.any(values <= 0):
            raise ValueError("LogModel encountered non-positive values.")
        return np.log(values)


class SqrtModel(Model):
    """Model representing np.sqrt(base_model output)."""

    def __init__(self, base: Model) -> None:
        if np.any(base.lower < 0):
            raise ValueError("SqrtModel requires base model output ≥ 0.")
        lower = np.sqrt(np.maximum(base.lower, 0))
        upper = np.sqrt(np.maximum(base.upper, 0))
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        super().__init__(lower, upper)
        self.base = base

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        super().evaluate_samples(samples)
        values = self.base.evaluate_samples(samples)
        if np.any(values < 0):
            raise ValueError("SqrtModel encountered negative values.")
        return np.sqrt(values)


class AbsModel(Model):
    """Model representing np.abs(base_model output)."""

    def __init__(self, base: Model) -> None:
        # Compute bounds conservatively
        lower = np.minimum(np.abs(base.lower), np.abs(base.upper))
        upper = np.maximum(np.abs(base.lower), np.abs(base.upper))
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        super().__init__(lower, upper)
        self.base = base

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate model and apply absolute value."""
        super().evaluate_samples(samples)
        values = self.base.evaluate_samples(samples)
        return np.abs(values)


class ExpNPModel(Model):
    """Model representing exp(base_model output)."""

    def __init__(self, base: Model) -> None:
        lower = np.exp(base.lower)
        upper = np.exp(base.upper)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        super().__init__(lower, upper)
        self.base = base

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(self, samples):
        super().evaluate_samples(samples)
        return np.exp(self.base.evaluate_samples(samples))


class SinModel(Model):
    """Model representing sin(base_model output)."""

    def __init__(self, base: Model) -> None:
        lower = np.sin(base.lower)
        upper = np.sin(base.upper)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        super().__init__(lower, upper)
        self.base = base

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate samples and apply sine."""
        super().evaluate_samples(samples)
        return np.sin(self.base.evaluate_samples(samples))


class CosModel(Model):
    """Model representing cos(base_model output)."""

    def __init__(self, base: Model) -> None:
        lower = np.cos(base.lower)
        upper = np.cos(base.upper)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        super().__init__(lower, upper)
        self.base = base

    @property
    def dof(self) -> int:
        return self.base.dof

    def evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        """Evaluate samples and apply cosine."""
        super().evaluate_samples(samples)
        return np.cos(self.base.evaluate_samples(samples))


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

    @classmethod
    def from_iter(cls, iterable):
        try:
            a, b = iterable  # exactly two models
        except ValueError:
            raise ValueError("Exactly 2 models required for division.") from None
        inst = super().from_iter([a, b])
        return inst

    def evaluate_samples(self, samples):
        super().evaluate_samples(samples)
        a_val = self._models[0].evaluate_samples(samples)
        b_val = self._models[1].evaluate_samples(samples)
        return a_val - b_val


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
    @classmethod
    def from_iter(cls, iterable):
        try:
            a, b = iterable  # exactly two models
        except ValueError:
            raise ValueError("Exactly 2 models required for division.") from None
        inst = cls.from_iter([a, b])
        return inst

    def evaluate_samples(self, samples):
        super().evaluate_samples(samples)
        a_val = self._models[0].evaluate_samples(samples)
        b_val = self._models[1].evaluate_samples(samples)
        return a_val / b_val
