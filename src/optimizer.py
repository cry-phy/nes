from functools import wraps
from typing import Callable
from dists.basedist import Dist
from models.basemodel import Model
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from typing import cast


class ConvergenceError(BaseException):
    """Exception to throw when an issue with convergence."""


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

        self._max_steps: int = 1000

    @property
    def tolerance(self) -> float:
        """The tolerance property."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value) -> None:
        self._tolerance = value

    @property
    def max_steps(self):
        """The max_steps property."""
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value):
        self._max_steps = value

    def gradient_decider(self) -> bool:
        """Decide continuation based on tolerance in gradient."""
        return self._gradient.sum() > self.tolerance

    def step(self) -> None:
        """Perform one step of optimization."""
        self.__samples = self.dist.get_samples(self.sample_num)
        self.__fitness = self.model.evaluate_samples(self.__samples)
        self.dist.param_variation(self.__samples, self.__fitness)

    def optimize(self) -> tuple[np.ndarray, float]:
        """Differentiate optimization from return logic."""
        for _ in range(self._max_steps):
            if self.decider():
                break
            else:
                self.step()
        else:
            raise ConvergenceError(f"Optimizer did not converge in {self._max_steps}.")
        opt_loc = self.dist.get_optimum()
        opt_fit = self.model.evaluate_samples(opt_loc)[0]
        return opt_loc, opt_fit


def req_axes(func):
    """Wrap functions requiring cls.ax object to be set."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
        return func(self, *args, **kwargs)

    return wrapper


class VizOptimizer(Optimizer):
    """Optimizer class directly plotting the progress of the optimization."""

    _ax: Axes | None = None
    _fig: Figure | SubFigure | None = None

    def __init__(self, model: Model, dist: Dist, sample_num: int) -> None:
        """Initialize VizOptimizer with new figure and Axes."""
        super().__init__(model, dist, sample_num)

    @classmethod
    def using_axis(
        cls, ax: Axes, model: Model, dist: Dist, sample_num: int
    ) -> "VizOptimizer":
        """Generate A Visualizer from an existing Axes instance."""
        instance = cls(model, dist, sample_num)
        instance.ax = ax
        instance.fig = ax.get_figure()
        return instance

    @property
    def ax(self):
        """The ax property."""
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value

    @property
    def fig(self):
        """The fig property."""
        return self._fig

    @fig.setter
    def fig(self, value):
        self._fig = value

    @req_axes
    def plot_samples(self):
        """Plot samples."""
        axes = cast(Axes, self.ax)
        self.model.plot_samples(axes, self.__samples)

    @req_axes
    def plot_fitness(self):
        """Plot fitness over the search area."""
        axes = cast(Axes, self.ax)
        self.model.plot_fitness(axes, self.model.lower, self.model.upper)

    @req_axes
    def step(self) -> None:
        """Extend the optimizer step function to plot the samples."""
        self.plot_samples()
        return super().step()
