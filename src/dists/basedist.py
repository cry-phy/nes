from models.basemodel import Model
import numpy as np


class Dist:
    """Base class for distributions."""

    def __init__(self, num_dimensions: int) -> None:
        """Generate Instance of Class."""
        self._num_dimensions = num_dimensions
        self._params = np.empty(0)
        self._learning_rate = 1

    @classmethod
    def from_dimensions(cls, num_dimensions: int) -> "Dist":
        """Generate a distribution from the number of dimensins of a Model."""
        raise NotImplementedError

    @classmethod
    def from_model(cls, model: Model) -> "Dist":
        """Generate a model from a Model directly."""
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        """Return the number of parameters of the Distribution."""
        return self._params.size

    @property
    def parameters(self):
        """The parameters property."""
        return self._params

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

    def get_samples(self, number: int) -> np.ndarray:
        """Get samples with current parameters."""
        raise NotImplementedError

    def param_variation(self, samples: np.ndarray, fitnesses: np.ndarray) -> None:
        """Get variation of parameters."""
        self._params += self._inverse_fisher(samples) @ self._gradient(
            samples, fitnesses
        )

    def set_init_params(self, *args, **kwargs) -> np.ndarray:
        """Get initial parameters."""
        raise NotImplementedError

    def get_optimum(self) -> np.ndarray:
        """Get optimum parameters."""
        raise NotImplementedError

    def _inverse_fisher(self, samples: np.ndarray) -> np.ndarray:
        """Compute inverse Fisher matrix.

        Depending on the case can be done closed form or via samples.
        """
        raise NotImplementedError

    def _gradient(self, samples: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Compute the gradient from the samples and fitnesses."""
        raise NotImplementedError
