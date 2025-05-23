"""Model module.

Provides access to Base and a few more Models.
"""

from .basemodel import Model
from .numericmodels import NPolynome, Spherical
from .composite import Composite

__all__ = [
    "Composite",
    "Model",
    "NPolynome",
    "Spherical",
]
