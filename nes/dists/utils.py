import numpy.typing as npt
import numpy as np


def grad_dep_rate(variation: npt.NDArray, alpha: float = 0.2) -> npt.NDArray:
    """Scale learning rate vector according to its 2 norm."""
    norm = np.linalg.norm(variation)
    eta = np.sqrt(1 + (alpha * norm) ** 2)
    return np.full_like(variation, eta)
