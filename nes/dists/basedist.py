from nes.models import Model
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse


class Dist(ABC):
    """Base class for distributions."""

    @abstractmethod
    def __init__(self, num_dimensions: int) -> None:
        """Generate Instance of Class.

        !!! Use factory methods instead !!!
        """
        if num_dimensions < 1:
            raise ValueError("Must be more than 0 dimensions.")
        self._num_dimensions: int = int(num_dimensions)
        self._params = np.empty(0, dtype=np.float64)
        self._learning_rate = 0.5

    @classmethod
    @abstractmethod
    def from_dimensions(cls, num_dimensions: int) -> "Dist":
        """Generate a distribution from the number of dimensins of a Model."""
        pass

    @classmethod
    @abstractmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model directly."""
        pass

    @property
    def num_params(self) -> int:
        """Return the number of parameters of the Distribution."""
        return self._params.size

    @property
    def parameters(self):
        """The parameters property."""
        return self._params.copy()

    @parameters.setter
    def parameters(self, value):
        self._params = value

    @property
    def learning_rate(self):
        """The learning_rate property."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @abstractmethod
    def get_samples(self, number: int) -> npt.NDArray[np.float64]:
        """Get samples with current parameters."""
        pass

    @abstractmethod
    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Get variation of parameters."""
        pass

    @abstractmethod
    def set_init_params(self, *args, **kwargs) -> npt.NDArray[np.float64]:
        """Get initial parameters."""
        pass

    @abstractmethod
    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum parameters."""
        pass

    def get_covariance(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        raise NotImplementedError(
            f"{__class__.__name__} does not implement a mean equivalent."
        )

    def get_mean(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        raise NotImplementedError(
            f"{__class__.__name__} does not implement a covarinance equivalent."
        )

    def plot_sigmas(
        self,
        ax: Axes,
        i1: int = 0,
        i2: int = 1,
        sigmas: tuple[float, ...] = (1, 2, 3, 5),
    ) -> None:
        """Display the fitness against the first two dimensions."""
        mu_2d = self.get_mean()[[i1, i2]]
        print(mu_2d, self.get_covariance())
        cov_2d = self.get_covariance()[np.ix_([i1, i2], [i1, i2])]
        print(mu_2d, cov_2d)
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        for s in sigmas:
            width, height = 2 * s * np.sqrt(eigvals)
            ell = Ellipse(
                xy=mu_2d,
                width=width,
                height=height,
                angle=angle,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                linewidth=1.0,
                label=f"{s}σ" if s == sigmas[0] else None,
            )
            ax.add_patch(ell)
