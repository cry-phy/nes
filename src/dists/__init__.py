"""Distribution module.

Provide Base and a few example Distributions.
"""

from .normal import DiagNormal
from .laplace import Laplace

__all__ = [
    "DiagNormal",
    "Laplace",
]
