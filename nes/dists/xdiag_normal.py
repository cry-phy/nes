from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt


class XDiagNormal(Dist):
    """Logarithmic representation of covariance parameters.

    Parameter layout:
        [mu_0, ..., mu_n, ln(sigma_0), ..., ln(sigma_n)]
    """

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        !!! Use factory methods instead !!!
        """
        super().__init__(num_dimensions)
        self._params: npt.NDArray[np.float64] = np.zeros(2 * num_dimensions)

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "Dist":
        """Generate empty Distribution."""
        return cls(num_dimensions)

    @classmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model directly."""
        lower, upper = model.search_range()
        instance = cls(len(lower))
        means = (upper + lower) / 2
        variances = 2 * np.log(upper - lower) + np.log(10)
        params = np.concatenate((means.reshape(-1), variances.reshape(-1)))
        instance.set_init_params(params)
        return instance

    @property
    def _mean(self) -> npt.NDArray[np.float64]:
        """The mean value."""
        return self._params[: self._num_dimensions].copy()

    @property
    def _variance(self) -> npt.NDArray[np.float64]:
        """The dimensional sigmas."""
        return np.exp(2 * self._params[self._num_dimensions :])

    def get_samples(self, number: int) -> npt.NDArray[np.float64]:
        """Generate samples."""
        z = np.random.randn(number, self._num_dimensions)
        return self._mean + z * np.sqrt(self._variance)

    def _gradient(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute gradient from samples."""
        gradients = np.zeros(shape=(len(samples), self.num_params))
        diffs = samples - self._mean
        gradients[:, : self._num_dimensions] = diffs / self._variance
        gradients[:, self._num_dimensions :] = diffs**2 / self._variance - 1
        adjusted_fitness = -fitnesses + fitnesses.max()
        weighted_gradients = gradients * (adjusted_fitness / adjusted_fitness.sum())
        return weighted_gradients.sum(axis=0)

    def _inverse_fisher(
        self, samples: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Analytically compute Fisher matrix."""
        invfisher = self._params.copy()
        invfisher[: self._num_dimensions] = self._variance
        invfisher[self._num_dimensions :] = 2.0
        return invfisher

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute standard Parameter variation."""
        variation = self._inverse_fisher(samples) * self._gradient(samples, fitnesses)
        self._params += variation
        return variation

    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum mean parameters."""
        return self._mean

    def set_init_params(self, parameters: npt.NDArray[np.float64]) -> None:
        """Get initial parameters."""
        if np.any(parameters[self._num_dimensions :] <= 0):
            raise ValueError("Cannot use negative variances.")
        self._params[: self._num_dimensions] = parameters[: self._num_dimensions]
        self._params[self._num_dimensions :] = np.log(
            parameters[self._num_dimensions :]
        )

    def get_covariance(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return np.diag(self._variance.copy())

    def get_mean(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._mean.copy()


class XDiagNormalSchaulLR(XDiagNormal):
    """Use Constant learning rates heuristically found by Schaul et al."""

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        !!! Use factory methods instead !!!
        """
        super().__init__(num_dimensions)
        mu_rates = np.full(self._num_dimensions, 1)
        var_rates = np.full(
            self._num_dimensions,
            (3 + np.log(self._num_dimensions)) / 5 / np.sqrt(self._num_dimensions),
        )
        self.rates = np.concatenate((mu_rates, var_rates))

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute standard Parameter variation."""
        variation = self._inverse_fisher(samples) * self._gradient(samples, fitnesses)
        update = self.rates * variation
        self._params += update
        return update


class XDiagNormalAdaptive(XDiagNormal):
    """Use KL-based adaptive learning rates with EMA stabilization."""

    klbudget: float = 0.1
    ema_beta: float = 0.9

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        !!! Use factory methods instead !!!
        """
        super().__init__(num_dimensions)
        mu_rates = np.full(self._num_dimensions, 1.0)
        var_rates = np.full(self._num_dimensions, 1.0)
        self.rates = np.concatenate((mu_rates, var_rates))
        self._grad_var_ema = 1.0

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute parameter variation with EMA-stabilized KL-based rate."""
        variation = self._inverse_fisher(samples) * self._gradient(samples, fitnesses)
        g_var = variation[self._num_dimensions :]
        norm_sq = np.sum(g_var**2)
        self._grad_var_ema = (
            self.ema_beta * self._grad_var_ema + (1 - self.ema_beta) * norm_sq
        )
        self.rates[self._num_dimensions :] = 2.0 * np.sqrt(
            self.klbudget / (self._grad_var_ema + 1e-12)
        )
        update = self.rates * variation
        self._params += update
        return update
