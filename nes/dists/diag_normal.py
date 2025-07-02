from nes.dists import Dist
from nes.models import Model
import numpy as np
import numpy.typing as npt


class DiagNormal(Dist):
    """Multivariate Normal distribution.

    Limited to diagonal covariance for now as there exists a closed form of cdf.
    """

    var_eps = 1e-8

    def __init__(self, num_dimensions: int) -> None:
        """Initialize instance of Normal distribution.

        !!! Use factory methods instead !!!
        """
        super().__init__(num_dimensions)
        self._params: npt.NDArray[np.float64] = np.concatenate(
            (
                np.zeros(num_dimensions, dtype=np.float64),
                np.ones(num_dimensions, dtype=np.float64) * 5,
            )
        )

    @property
    def _mean(self) -> npt.NDArray[np.float64]:
        """The mean value."""
        return self._params[: self._num_dimensions].copy()

    @property
    def _variance(self) -> npt.NDArray[np.float64]:
        """The dimensional sigmas."""
        return self._params[self._num_dimensions :].copy()

    def safe_variance(self) -> npt.NDArray[np.float64]:
        """Limit variance to positive values."""
        return np.maximum(self._variance, self.var_eps)

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "DiagNormal":
        """Generate Normal distribution with number of dimensions."""
        return cls(num_dimensions)

    @classmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model directly."""
        lower, upper = model.search_range()
        instance = cls(len(lower))
        means = (upper + lower) / 2
        variances = ((upper - lower) / 6) ** 2 / 10
        params = np.concatenate((means.reshape(-1), variances.reshape(-1)))
        instance.set_init_params(params)
        return instance

    def get_samples(self, number: int) -> npt.NDArray[np.float64]:
        """Generate samples."""
        z = np.random.randn(number, self._num_dimensions)
        return self._mean + z * np.sqrt(self._variance)

    def _gradient(self, samples: npt.NDArray, fitnesses: npt.NDArray) -> npt.NDArray:
        """Compute gradient from samples."""
        gradients = np.zeros(shape=(len(samples), self.num_params))
        diffs = samples - self._mean
        gradients[:, : self._num_dimensions] = diffs / self.safe_variance()
        gradients[:, self._num_dimensions :] = (
            0.5 * (diffs**2 / self.safe_variance() - 1) / self.safe_variance()
        )
        adjusted_fitness = -fitnesses + fitnesses.max()
        weighted_gradients = gradients * (adjusted_fitness / adjusted_fitness.sum())
        return weighted_gradients.sum(axis=0)

    def _inverse_fisher(self, samples: npt.NDArray) -> npt.NDArray:
        """Compute Fisher matrix.

        Since there exists a closed form we simply use it here.
        """
        invfisher = self._params.copy()
        invfisher[: self._num_dimensions] = invfisher[self._num_dimensions :]
        invfisher[self._num_dimensions :] = 2 * invfisher[self._num_dimensions :] ** 2
        return invfisher

    def _analytical_fisher(self) -> npt.NDArray[np.float64]:
        """Compute analytical Fisher matrix."""
        fisher = np.zeros(self.num_params, dtype=np.float64)
        fisher[: self._num_dimensions] = 1 / self.safe_variance()
        fisher[self._num_dimensions :] = 1 / (2 * self.safe_variance() ** 2)
        return fisher

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Get variation of parameters."""
        variation = self._inverse_fisher(samples) * self._gradient(samples, fitnesses)
        self._params += variation
        self._params[self._num_dimensions :] = np.maximum(
            self._params[self._num_dimensions :], self.var_eps
        )
        return variation

    def get_optimum(self) -> npt.NDArray[np.float64]:
        """Get optimum parameters."""
        return self._mean.copy()

    def set_init_params(self, parameters: npt.NDArray[np.float64]) -> None:
        """Get initial parameters."""
        self._params = parameters

    def get_covariance(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return np.diag(self._variance.copy())

    def get_mean(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._mean.copy()


class DiagNormalSchaulLR(DiagNormal):
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


class DiagNormalAdaptive(DiagNormal):
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
