
import numpy as np
import numpy.typing as npt
from nes.dists import Dist, DiagNormal
from nes.models import NPolynome
from nes.optimizer import Optimizer
# from nes.models.benchmarks2D import *
import matplotlib.pyplot as plt

def convergence(model, dist: Dist, sample_num: int, n_steps = 1000, initial_guess = None) -> npt.NDArray[np.float64]:
    """If initial guess is defined must be a the shape of the distribution
    parameters of the given distribution"""
    if initial_guess is not None:
        dist.set_init_params(initial_guess)

    error = np.zeros(n_steps)
    optimizer = Optimizer(model, dist, sample_num)
    for i in range(n_steps):
        optimizer.step()
        error[i] = np.linalg.norm(optimizer.dist.get_optimum() - model.optimum)
    return error


def plot_dims_analysis(dims, Distributions, sample_num = 100, n_steps = 1000, initial_guess=None):
    errors = []
    for Distribution in Distributions:
        for dim in dims:
            weights = np.array([[0]*dim, [0]*dim, [1.]*dim])
            model = NPolynome(weights)
            model.lower = np.full(weights.shape[1], -20)
            model.upper = np.full(weights.shape[1], +20)
            dist = Distribution.from_model(model)
            if initial_guess is not None:
                dist.set_init_params(initial_guess)
            optimizer = Optimizer(model, dist, sample_num)
            for _ in range(n_steps):
                optimizer.step()
            errors.append(np.linalg.norm(dist.get_optimum()))
    plt.plot(dims, errors)
    plt.show()

dims = np.arange(1, 101, 4, dtype=int)
plot_dims_analysis(dims, [DiagNormal])


# my_model = Rastrigin.model()
# my_conv = convergence(my_model, DiagNormal.from_model(my_model), 50)
# plt.plot(my_conv)
# plt.show()
