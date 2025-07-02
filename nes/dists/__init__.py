"""Distribution module.

Provide Base and a few example Distributions.
"""

from .basedist import Dist
from .laplace import Laplace
from .diag_normal import DiagNormal, DiagNormalSchaulLR, DiagNormalAdaptive
from .xdiag_normal import XDiagNormal, XDiagNormalSchaulLR, XDiagNormalAdaptive
from .student_t import StudentTDist, StudentTDistSchaulLR, StudentTDistAdaptive
from .xnormal import XNormal, XNormalSchaulRL, XNormalAdaptive
from .normal import Normal

__all__ = [
    "DiagNormal",
    "DiagNormalAdaptive",
    "DiagNormalSchaulLR",
    "Dist",
    "Laplace",
    "Normal",
    "StudentTDist",
    "StudentTDistAdaptive",
    "StudentTDistSchaulLR",
    "XDiagNormal",
    "XDiagNormalAdaptive",
    "XDiagNormalSchaulLR",
    "XNormal",
    "XNormalAdaptive",
    "XNormalSchaulRL",
]
