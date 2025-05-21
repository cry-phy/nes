"""Distribution module.

Provide Base and a few example Distributions.
"""

from .basedist import Dist
from .normal import DiagNormal
from .laplace import Laplace

__all__ = [
    "DiagNormal",
    "Dist",
    "Laplace",
]
