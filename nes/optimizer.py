from collections import deque
from functools import wraps
from typing import Callable
from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from typing import cast


class ConvergenceError(BaseException):
    """Exception to throw when an issue with convergence."""


class Optimizer:
    """Optimizer class orchestrating the Model and Distribution."""

    def __init__(
        self, model: Model, dist: Dist, sample_num: int, maxlen: int = 10
    ) -> None:
        """Initialize Optimizer.

        Set with preset Model and distribution.
        """
        self.model: Model = model
        self.dist: Dist = dist

        self.sample_num = sample_num
        self._max_steps: int = int(1e4)

        self._tolerance: float = 1e-3
        self.decider: Callable[[], bool] = self.gradient_decider

        self._hlen: int = maxlen
        self._prev_samples: deque[npt.NDArray[np.float64]] = deque(maxlen=self._hlen)
        self._prev_fitnesses: deque[npt.NDArray[np.float64]] = deque(maxlen=self._hlen)
        self._prev_variation: deque[npt.NDArray[np.float64]] = deque(maxlen=self._hlen)
        self._prev_parameter: deque[npt.NDArray[np.float64]] = deque(maxlen=self._hlen)

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
        grads = np.array(self._prev_variation)
        norms = np.linalg.norm(grads, axis=1)
        return norms.mean() <= self.tolerance

    def diff_decider(self) -> bool:
        """Decide continuation based on tolerance in gradient."""
        prev_diff = self._prev_samples[0].mean(axis=0) - self._prev_samples[1].mean(
            axis=0
        )
        return np.sqrt((prev_diff**2).sum()) < self.tolerance

    def step(self) -> None:
        """Perform one step of optimization."""
        self._prev_parameter.appendleft(self.dist.parameters)

        samples = self.dist.get_samples(self.sample_num)
        fitness = self.model.evaluate_samples(samples)
        variation = self.dist.param_variation(samples, fitness)

        self._prev_samples.appendleft(samples)
        self._prev_fitnesses.appendleft(fitness)
        self._prev_variation.appendleft(variation)

    def optimize(self) -> tuple[np.ndarray, float]:
        """Differentiate optimization from return logic."""
        self.step()
        for n in range(self._max_steps):
            if self.decider():
                print(f"Optimized afer {n} steps!")
                break
            else:
                #  print(f"Step {n}: {self.dist.parameters}")
                self.step()
        else:
            raise ConvergenceError(f"Optimizer did not converge in {self._max_steps}.")
        opt_loc = self.dist.get_optimum().reshape((1, self.model.dof))
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
        super().__init__(model, dist, sample_num, maxlen=int(1e5))

        if self.model.dof != 2:
            raise Warning("Cannot use direction plotting for other than 2D,")

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
    def plot_samples(self, max_samples: int = 32):
        """Plot samples."""
        axes = cast(Axes, self.ax)
        self.model.plot_samples(
            axes,
            self._prev_samples[0][:max_samples],
        )

    @req_axes
    def plot_variation(self):
        """Plot variation direction."""
        axes = cast(Axes, self.ax)
        mean = self._prev_samples[0][:2].mean(axis=0)
        dxs = self._prev_variation[0][:2]
        axes.annotate("", xytext=mean, xy=mean + dxs, arrowprops={"arrowstyle": "->"})

    @req_axes
    def plot_fitness(self):
        """Plot fitness over the search area."""
        axes = cast(Axes, self.ax)
        self.model.plot_fitness(axes, self.model.lower, self.model.upper)

    @req_axes
    def step(self, show: bool = False) -> None:
        """Extend the optimizer step function to plot the samples."""
        super().step()
        if show:
            self.plot_samples()

    @req_axes
    def plot_series(self) -> None:
        """Plot parameters over time."""
        axes = cast(Axes, self.ax)
        axes_right = plt.twinx(axes)
        data = np.flipud(np.array(self._prev_parameter))
        axes.plot(data[:, 0], label="x")
        axes.plot(data[:, 1], label="y")
        axes_right.plot(data[:, 2], label="var x", alpha=0.3)
        axes_right.plot(data[:, 3], label="var y", alpha=0.3)
        axes.legend()
