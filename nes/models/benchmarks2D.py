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
        model = (
            10 * 2
            + x_**2
            - 10 * np.cos(2 * np.pi * x_)
            + y_**2
            - 10 * np.cos(2 * np.pi * y_)
        )
        model.lower = np.array([-5.12, -5.12])
        model.upper = np.array([5.12, 5.12])
        model.optimum = np.array([0., 0.])
        return model


class Ackley:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (
            -20 * np.exp(-0.2 * (0.5 * (x_**2 + y_**2)) ** 0.5)
            - np.exp(0.5 * (np.cos(2 * np.pi * x_) + np.cos(2 * np.pi * y_)))
            + np.e
            + 20
        )
        model.lower = np.array([-32.768, -32.768])
        model.upper = np.array([32.768, 32.768])
        model.optimum = np.array([0., 0.])

        return model


class Sphere:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = x_**2 + y_**2
        model.lower = np.array([-5.12, -5.12])
        model.upper = np.array([5.12, 5.12])
        model.optimum = np.array([0., 0.])

        return model


class Rosenbrock:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = 100 * (y_ - x_**2) ** 2 + (1 - x_) ** 2
        model.lower = np.array([-2.048, -2.048])
        model.upper = np.array([2.048, 2.048])
        model.optimum = np.array([1., 1.])

        return model


class Beale:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (
            (1.5 - x_ + x_ * y_) ** 2
            + (2.25 - x_ + x_ * y_**2) ** 2
            + (2.625 - x_ + x_ * y_**3) ** 2
        )
        model.lower = np.array([-4.5, -4.5])
        model.upper = np.array([4.5, 4.5])
        model.optimum = np.array([3., 0.5])

        return model


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
        model = term1 * term2
        model.lower = np.array([-2, -2])
        model.upper = np.array([2, 2])
        model.optimum = np.array([3., -1.])
        return model


class Booth:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (x_ + 2 * y_ - 7) ** 2 + (2 * x_ + y_ - 5) ** 2
        model.lower = np.array([-10, -10])
        model.upper = np.array([10, 10])
        model.optimum = np.array([1., 3.])
        return model


class BukinN6:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = 100 * (np.abs(y_ - 0.01 * x_**2) ** 0.5) + 0.01 * np.abs(x_ + 10)
        model.lower = np.array([-15, -3])
        model.upper = np.array([-5, 3])
        model.optimum = np.array([-10., 1.])
        return model


class Matyas:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = 0.26 * (x_**2 + y_**2) - 0.48 * x_ * y_
        model.lower = np.array([-10, -10])
        model.upper = np.array([10, 10])
        model.optimum = np.array([0., 0.])
        return model


class Levi13:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (
            np.sin(3 * np.pi * x_) ** 2
            + (x_ - 1) ** 2 * (1 + np.sin(3 * np.pi * y_) ** 2)
            + (y_ - 1) ** 2 * (1 + np.sin(2 * np.pi * y_) ** 2)
        )
        model.lower = np.array([-10, -10])
        model.upper = np.array([10, 10])
        model.optimum = np.array([1., 1.])
        return model


class Griewank:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (
            1
            + (x_**2 + y_**2) / 4000
            - np.cos(x_ / np.sqrt(1)) * np.cos(y_ / np.sqrt(2))
        )
        model.lower = np.array([-100, -100])  # Adjusted from [-600, -600]
        model.upper = np.array([100, 100])   # Adjusted from [600, 600]
        model.optimum = np.array([0., 0.])
        return model


class Himmelblau:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (x_**2 + y_ - 11) ** 2 + (x_ + y_**2 - 7) ** 2
        model.lower = np.array([-5, -5])
        model.upper = np.array([5, 5])
        return model


class ThreeHumpCamel:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = 2 * x_**2 - 1.05 * x_**4 + (x_**6) / 6 + x_ * y_ + y_**2
        model.lower = np.array([-5, -5])
        model.upper = np.array([5, 5])
        model.optimum = np.array([0., 0.])
        return model


class Easom:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (
            -np.cos(x_) * np.cos(y_) * np.exp(-((x_ - np.pi) ** 2 + (y_ - np.pi) ** 2))
        )
        model.lower = np.array([-50, -50])  # Adjusted from [-100, -100]
        model.upper = np.array([50, 50])   # Adjusted from [100, 100]
        model.optimum = np.array([np.pi, np.pi])
        return model


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
        model = -0.0001 * expr**0.1
        model.lower = np.array([-10, -10])
        model.upper = np.array([10, 10])
        model.optimum = np.array([0., 0.])
        return model


class Eggholder:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = (
            -(y_ + 47) * np.sin(np.sqrt(np.abs(y_ + x_ / 2 + 47)))
            - x_ * np.sin(np.sqrt(np.abs(x_ - (y_ + 47))))
        )
        model.lower = np.array([-512, -512])
        model.upper = np.array([512, 512])
        model.optimum = np.array([512., 404.2319])
        return model


class HolderTable:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = -np.abs(
            np.sin(x_) * np.cos(y_) * np.exp(np.abs(1 - (np.sqrt(x_**2 + y_**2) / np.pi)))
        )
        model.lower = np.array([-10, -10])
        model.upper = np.array([10, 10])
        return model


class McCormick:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = np.sin(x_ + y_) + (x_ - y_) ** 2 - 1.5 * x_ + 2.5 * y_ + 1
        model.lower = np.array([-1.5, -3])
        model.upper = np.array([4, 4])
        return model


class StyblinskiTang:
    @staticmethod
    def model():
        x_ = deepcopy(_x)
        y_ = deepcopy(_y)
        model = 0.5 * ((x_**4 - 16 * x_**2 + 5 * x_) + (y_**4 - 16 * y_**2 + 5 * y_))
        model.lower = np.array([-5, -5])
        model.upper = np.array([5, 5])
        model.optimum = np.array([2.903534, 2.903534])
        return model


__all__ = [
    "Rastrigin",
    "Ackley",
    "Sphere",
    "Rosenbrock",
    "Beale",
    "GoldsteinPrice",
    "Booth",
    "BukinN6",
    "Matyas",
    "Levi13",
    "Griewank",
    "Himmelblau",
    "ThreeHumpCamel",
    "Easom",
    "CrossInTray",
    "Eggholder",
    "HolderTable",
    "McCormick",
    "StyblinskiTang",
]
