from models.basemodel import Model
import numpy as np
from matplotlib.axes import Axes


class NPolynome(Model):
    """d->1 dimensional Polynome.

    Contains:
    - Optimum Location
    - Directional scalings
    """

    def __init__(self, weights: np.ndarray[np.float64]) -> None:
        """Initialize high dimensional parabola.

        Weights w up to max order n and dimension d are structured:

        w00, w01, ..., w0d
        w10, w11, ..., w1d
        ..., ..., ..., ...
        wn0, wn0, ..., wnd

        """
        self.weights: np.ndarray[np.float64] = weights
        self.num_dims: int = weights.shape[1]

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        return self.num_dims

    def evaluate_samples(
        self, samples: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Evaluate list of samples and return fitness in order.

        Broadcasting is a bitch. Counter: 2.5h
        """
        orders = np.arange(self.weights.shape[0])
        weights_expanded = self.weights.T[np.newaxis, :, :]
        exponentiated = samples[:, :, np.newaxis] ** orders[np.newaxis, np.newaxis, :]
        return (weights_expanded * exponentiated).sum(axis=(1, 2))


class Spherical(Model):
    """Real Part Spherical Harmonic."""

    def __init__(self, l_order: int, m_order: int) -> None:
        """Initialize Spherical Harmonic of order l, m."""
        super().__init__()
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
            * np.special.factorial(self.l - self.m)
            / np.special.factorial(self.l + self.m)
        )
        self.__coeff2 = (-1) ** self.m * 2**self.l
        ks = np.arange(self.m, self.l + 1)
        self.__coeff3 = (
            np.special.factorial(ks)
            / np.special.factorial(ks - self.m)
            * np.special.binom(self.l, ks)
            * np.special.binom((self.l + ks - 1) / 2, self.l)
        )

    def __ass_legendre(self, values: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Evaluate associated Legendre Polynome."""
        return (
            self.__coeff2
            * (1 - values) ** (self.m / 2)
            * self.polynome.evaluate_samples(values)
        )

    @property
    def dof(self) -> int:
        """The degrees_of_freedom property."""
        return 2

    def evaluate_samples(
        self, samples: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Evaluate list of samples and return fitness in order."""
        return (
            self.__coeff1
            * self.__ass_legendre(np.cos(samples[:, 1]))
            * np.cos(self.m * samples[:, 0])
        )

    def initial_guess(self) -> np.ndarray:
        """Make initial guess for sample."""
        raise NotImplementedError

    def plot_fitness(
        self,
        ax: Axes,
        lower: np.ndarray[np.float64],
        upper: np.ndarray[np.float64],
    ) -> None:
        """Display the fitness."""
        n: int = 101
        phis: np.ndarray = np.linspace(lower[0], upper[0], n)
        thetas: np.ndarray = np.linspace(lower[1], upper[1], n)
        x, y = np.meshgrid(phis, thetas)
        samples = np.array(x, y).reshape(2, -1).T
        fitness = self.evaluate_samples(samples)
        ax.contourf(x, y, fitness, levels=50, cmap="")

    def plot_samples(self, ax: Axes, samples: np.ndarray[np.float64]) -> None:
        """Display the samples."""
        phis: np.ndarray = samples[:, 0]
        thetas: np.ndarray = samples[:, 1]
        ax.scatter(phis, thetas, s=40, facecolors="none", edgecolors="b", cmap="plasma")
