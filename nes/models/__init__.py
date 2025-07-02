"""Model module.

Provides access to Base and a few more Models.
"""

from .basemodel import Model, Composite
from .numericmodels import NPolynome, Spherical
from . import benchmarks2D

__all__ = [
    "benchmarks2D",
    "Composite",
    "Model",
    "NPolynome",
    "Spherical",
]
