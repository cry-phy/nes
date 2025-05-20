"""Model module.

Provides access to Base and a few more Models.
"""

from .basemodel import Model
from .numericmodels import NPolynome, Spherical

__all__ = [
    "Model",
    "NPolynome",
    "Spherical",
]
