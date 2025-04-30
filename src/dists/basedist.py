from models.basemodel import Model
import numpy as np


class Dist:
    """Base class for distributions."""

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "Dist":
        """Generate a distribution from the number of dimensins of a Model."""
        raise NotImplementedError

    @classmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model diretly."""
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        """Return the number of parameters of the Distribution."""
        raise NotImplementedError

    def get_samples(self, parameters: np.ndarray) -> np.ndarray:
        """Get samples with current parameters."""
        raise NotImplementedError

    def get_param_variation(
        self, samples: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        """Get variation of parameters."""
        return self._inverse_fisher(samples) @ self._gradient(samples, fitnesses)

    def get_init_params(self, *args, **kwargs) -> np.ndarray:
        """Get initial parameters."""
        raise NotImplementedError

    def _inverse_fisher(self, samples: np.ndarray) -> np.ndarray:
        """Compute inverse Fisher matrix.

        Depending on the case can be done closed form or via samples.
        """

    def _gradient(self, samples: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Compute the gradient from the samples and fitnesses."""
