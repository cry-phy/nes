from models import Model
import numpy as np
import numpy.typing as npt
from scipy import special
from matplotlib.axes import Axes


class NPolynome(Model):
    """d->1 dimensional Polynome.

    Contains:
    - Optimum Location
    - Directional scalings
    """

    def __init__(
        self,
        weights: npt.NDArray[np.float64],
    ) -> None:
        """Initialize high dimensional parabola.

        Weights w up to max order n and dimension d are structured:

        w00, w01, ..., w0d
        w10, w11, ..., w1d
        ..., ..., ..., ...
        wn0, wn0, ..., wnd

        """
        lower = np.full(weights.shape[1], -np.inf)
        upper = np.full(weights.shape[1], np.inf)
        super().__init__(lower, upper)
        self.weights: npt.NDArray[np.float64] = weights
        self.num_dims: int = weights.shape[1]

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        return self.num_dims

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate list of samples and return fitness in order.

        Broadcasting is a bitch. Counter: 3h.
        """
        orders = np.arange(self.weights.shape[0])
        weights_expanded = self.weights.T[np.newaxis, :, :]
        exponentiated = samples[:, :, np.newaxis] ** orders[np.newaxis, np.newaxis, :]
        return (weights_expanded * exponentiated).sum(axis=(1, 2)).reshape(-1, 1)

    def plot_fitness(
        self,
        ax: Axes,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
    ) -> None:
        """Display the fitness against the first two dimensions."""
        n: int = 101
        xs: npt.NDArray = np.linspace(lower[0], upper[0], n)
        ys: npt.NDArray = np.linspace(lower[1], upper[1], n)
        x, y = np.meshgrid(xs, ys)
        samples = np.array([x, y]).reshape(2, -1).T
        fitness = self.evaluate_samples(samples).reshape(x.shape)
        ax.contourf(x, y, fitness, levels=50, cmap="plasma")

    def plot_samples(self, ax: Axes, samples: npt.NDArray[np.float64]) -> None:
        """Display the first two dimensions of samples."""
        phis: npt.NDArray = samples[:, 0]
        thetas: npt.NDArray = samples[:, 1]
        ax.scatter(phis, thetas, s=40, facecolors="none", edgecolors="b")


class Spherical(Model):
    """Real Part Spherical Harmonic.

    2 DOF: [Phi, Theta]
    """

    def __init__(self, l_order: int, m_order: int) -> None:
        """Initialize Spherical Harmonic of order l, m."""
        if l_order < m_order:
            raise ValueError("Order l must be larger than order m")
        lower = np.array([0, 0])
        upper = np.array([2 * np.pi, np.pi])
        super().__init__(lower, upper)
        self.l: int = l_order
        self.m: int = m_order

        self.__gen_coefficients()
        self.polynome: NPolynome = NPolynome(
            weights=self.__coeff3.reshape(self.l - self.m + 1, 1)
        )

    def __gen_coefficients(self) -> None:
        """Generate Coefficients for the associated Legendre Polynome."""
        self.__coeff1 = (-1) ** self.m * np.sqrt(
            (2 * self.l + 1)
            / (4 * np.pi)
            * special.factorial(self.l - self.m)
            / special.factorial(self.l + self.m)
        )
        self.__coeff2 = (-1) ** self.m * 2**self.l
        ks = np.arange(self.m, self.l + 1)
        self.__coeff3 = (
            special.factorial(ks)
            / special.factorial(ks - self.m)
            * special.binom(self.l, ks)
            * special.binom((self.l + ks - 1) / 2, self.l)
        )

    def __ass_legendre(
        self, values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate associated Legendre Polynome."""
        return (
            self.__coeff2
            * (1 - values**2) ** (self.m / 2)
            * self.polynome.evaluate_samples(values)
        )

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        return 2

    def evaluate_samples(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate list of samples and return fitness in order."""
        return (
            self.__coeff1
            * self.__ass_legendre(np.cos(samples[:, 1]).reshape(-1, 1))
            * np.cos(2 * self.m * samples[:, 0].reshape(-1, 1))
        )

    def plot_fitness(
        self,
        ax: Axes,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
    ) -> None:
        """Display the fitness."""
        n: int = 101
        phis: npt.NDArray = np.linspace(lower[0], upper[0], n)
        thetas: npt.NDArray = np.linspace(lower[1], upper[1], n)
        x, y = np.meshgrid(phis, thetas)
        samples = np.array([x, y]).reshape(2, -1).T
        fitness = self.evaluate_samples(samples).reshape(x.shape)
        ax.contourf(x, y, fitness, levels=50, cmap="plasma")

    def plot_samples(self, ax: Axes, samples: npt.NDArray[np.float64]) -> None:
        """Display the samples."""
        phis: npt.NDArray = samples[:, 0]
        thetas: npt.NDArray = samples[:, 1]
        ax.scatter(phis, thetas, s=40, facecolors="none", edgecolors="b")
