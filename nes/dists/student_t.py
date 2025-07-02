from nes.dists import Dist
import numpy as np
import numpy.typing as npt
from scipy.stats import chi2
from nes.models import Model
from scipy.special import psi


class StudentTDist(Dist):
    """Heavy-tailed distribution using diagonal multivariate Student-t."""

    def __init__(self, num_dimensions: int) -> None:
        super().__init__(num_dimensions)
        self._params = np.concatenate(
            [np.zeros(num_dimensions), np.ones(num_dimensions), np.ones(1)]
        )

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "StudentTDist":
        return cls(num_dimensions)

    @classmethod
    def from_model(cls, model: Model) -> "StudentTDist":
        return cls(model.dof)

    @property
    def _mean(self):
        """The _means property."""
        return self._params[: self._num_dimensions]

    @property
    def _stddev(self):
        """The _stddevs property."""
        return np.exp(self._params[self._num_dimensions : 2 * self._num_dimensions])

    @property
    def _nu(self):
        """The _stddevs property."""
        return np.exp(self._params[-1])

    def set_init_params(
        self, params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        self._params = params
        return self._params

    def get_optimum(self) -> npt.NDArray[np.float64]:
        return self._mean.copy()

    def get_samples(self, number: int) -> npt.NDArray[np.float64]:
        z = np.random.randn(number, self._num_dimensions)
        u = chi2.rvs(df=self._nu, size=number).reshape(-1, 1)
        scaled = z / np.sqrt(u / self._nu)
        return self._mean + self._stddev * scaled

    def _gradient(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        n, d = samples.shape
        mean = self._mean
        std = self._stddev
        nu = self._nu
        z = (samples - mean) / std
        norm_sq = np.sum(z**2, axis=1, keepdims=True)

        # --- Score function mean
        dlogp_dmu = ((nu + d) * z) / ((nu + norm_sq) * std)

        # --- Score function log std
        dlogp_dlogstd = ((nu + d) * z**2) / (nu + norm_sq) - 1

        # --- Score function log nu
        psi_diff = psi((nu + d) / 2) - psi(nu / 2)
        term1 = 0.5 * psi_diff
        term2 = -0.5 * np.log(1 + norm_sq / nu)
        term3 = -0.5 * d / nu
        term4 = 0.5 * (nu + d) * norm_sq / (nu**2 + nu * norm_sq)
        dlogp_dlognu = nu * (term1 + term2 + term3 + term4)

        # --- Fitness-weighted gradients
        ranks = np.argsort(np.argsort(fitnesses.flatten()))
        inv_ranks = n - 1 - ranks
        weights = inv_ranks.reshape(-1, 1) / inv_ranks.sum()
        grad_mu = np.mean(weights * dlogp_dmu, axis=0)
        grad_logstd = np.mean(weights * dlogp_dlogstd, axis=0)
        grad_lognu = np.mean(weights * dlogp_dlognu, axis=0)

        return np.concatenate([grad_mu, grad_logstd, grad_lognu.ravel()])

    def _inverse_fisher(self, samples: np.ndarray) -> np.ndarray:
        n, d = samples.shape
        mean = self._mean
        std = self._stddev
        nu = self._nu
        z = (samples - mean) / std
        norm_sq = np.sum(z**2, axis=1, keepdims=True)

        # Mean component
        dlogp_dmu = ((nu + d) * z) / ((nu + norm_sq) * std)
        fisher_mu = np.mean(dlogp_dmu**2, axis=0)

        # Log std component
        dlogp_dlogstd = ((nu + d) * z**2) / (nu + norm_sq) - 1
        fisher_logstd = np.mean(dlogp_dlogstd**2, axis=0)

        # Log nu component
        psi_diff = psi((nu + d) / 2) - psi(nu / 2)
        term1 = 0.5 * psi_diff
        term2 = -0.5 * np.log(1 + norm_sq / nu)
        term3 = -0.5 * d / nu
        term4 = 0.5 * (nu + d) * norm_sq / (nu**2 + nu * norm_sq)
        dlogp_dlognu = nu * (term1 + term2 + term3 + term4)
        fisher_lognu = np.mean(dlogp_dlognu**2, axis=0)

        fisher_diag = np.concatenate([fisher_mu, fisher_logstd, fisher_lognu.ravel()])
        return 1 / (fisher_diag + 1e-8)

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        fitnesses = fitnesses.reshape(-1, 1)
        grad = self._gradient(samples, fitnesses)
        fisher_inv = self._inverse_fisher(samples)
        variation = fisher_inv * grad
        self._params += -variation
        return variation

    def get_covariance(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return np.diag(self._stddev**2)

    def get_mean(self) -> npt.NDArray[np.float64]:
        """Get a covarinance matrix."""
        return self._mean.copy()


class StudentTDistSchaulLR(StudentTDist):
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
        nu_rates = np.full(1, 1)
        self.rates = np.concatenate((mu_rates, var_rates, nu_rates))

    def param_variation(
        self, samples: npt.NDArray[np.float64], fitnesses: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute standard Parameter variation."""
        variation = self._inverse_fisher(samples) * self._gradient(samples, fitnesses)
        update = self.rates * variation
        self._params += update
        return update


class StudentTDistAdaptive(StudentTDist):
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
        nu_rates = np.full(1, 1.0)
        self.rates = np.concatenate((mu_rates, var_rates, nu_rates))
        self._grad_var_ema = 0.5

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
        self.rates[self._num_dimensions : 2 * self._num_dimensions] = 2.0 * np.sqrt(
            self.klbudget / (self._grad_var_ema + 1e-12)
        )
        update = self.rates * variation
        self._params += update
        return update
