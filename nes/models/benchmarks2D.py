from nes.models import NPolynome
from copy import deepcopy
import numpy as np

_x = NPolynome(np.array([[0, 0], [1, 0]]))
_y = NPolynome(np.array([[0, 0], [0, 1]]))


class Rastrigin:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (
            10 * 2
            + x_**2
            - 10 * np.cos(2 * np.pi * x_)
            + y_**2
            - 10 * np.cos(2 * np.pi * y_)
        )


class Ackley:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (
            -20 * np.exp(-0.2 * (0.5 * (x_**2 + y_**2)) ** 0.5)
            - np.exp(0.5 * (np.cos(2 * np.pi * x_) + np.cos(2 * np.pi * y_)))
            + np.e
            + 20
        )


class Sphere:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return x_**2 + y_**2


class Rosenbrock:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return 100 * (y_ - x_**2) ** 2 + (1 - x_) ** 2


class Beale:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (
            (1.5 - x_ + x_ * y_) ** 2
            + (2.25 - x_ + x_ * y_**2) ** 2
            + (2.625 - x_ + x_ * y_**3) ** 2
        )


class GoldsteinPrice:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        term1 = 1 + (x_ + y_ + 1) ** 2 * (
            19 - 14 * x_ + 3 * x_**2 - 14 * y_ + 6 * x_ * y_ + 3 * y_**2
        )
        term2 = 30 + (2 * x_ - 3 * y_) ** 2 * (
            18 - 32 * x_ + 12 * x_**2 + 48 * y_ - 36 * x_ * y_ + 27 * y_**2
        )
        return term1 * term2


class Booth:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (x_ + 2 * y_ - 7) ** 2 + (2 * x_ + y_ - 5) ** 2


class BukinN6:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return 100 * (np.abs(y_ - 0.01 * x_**2) ** 0.5) + 0.01 * np.abs(x_ + 10)


class Matyas:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return 0.26 * (x_**2 + y_**2) - 0.48 * x_ * y_


class Levi13:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (
            np.sin(3 * np.pi * x_) ** 2
            + (x_ - 1) ** 2 * (1 + np.sin(3 * np.pi * y_) ** 2)
            + (y_ - 1) ** 2 * (1 + np.sin(2 * np.pi * y_) ** 2)
        )


class Griewank:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (
            1
            + (x_**2 + y_**2) / 4000
            - np.cos(x_ / np.sqrt(1)) * np.cos(y_ / np.sqrt(2))
        )


class Himmelblau:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (x_**2 + y_ - 11) ** 2 + (x_ + y_**2 - 7) ** 2


class ThreeHumpCamel:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return 2 * x_**2 - 1.05 * x_**4 + (x_**6) / 6 + x_ * y_ + y_**2


class Easom:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return (
            -np.cos(x_) * np.cos(y_) * np.exp(-((x_ - np.pi) ** 2 + (y_ - np.pi) ** 2))
        )


class CrossInTray:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        expr = (
            np.abs(
                np.sin(x_)
                * np.sin(y_)
                * np.exp(np.abs(100 - (x_**2 + y_**2) ** 0.5 / np.pi))
            )
            + 1
        )
        return -0.0001 * expr**0.1


class Eggholder:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return -(y_ + 47) * np.sin(np.sqrt(np.abs(x_ / 2 + y_ + 47))) - x_ * np.sin(
            np.sqrt(np.abs(x_ - (y_ + 47)))
        )


class HolderTable:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return -np.abs(
            np.sin(x_) * np.cos(y_) * np.exp(np.abs(1 - (x_**2 + y_**2) ** 0.5 / np.pi))
        )


class McCormick:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return np.sin(x_ + y_) + (x_ - y_) ** 2 - 1.5 * x_ + 2.5 * y_ + 1


class StyblinskiTang:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        return 0.5 * ((x_**4 - 16 * x_**2 + 5 * x_) + (y_**4 - 16 * y_**2 + 5 * y_))
