"""Model module.

Provides access to Base and a few more Models.
"""

from .basemodel import Model, Composite
from .numericmodels import NPolynome, Spherical

__all__ = [
    "Composite",
    "Model",
    "NPolynome",
    "Spherical",
]
