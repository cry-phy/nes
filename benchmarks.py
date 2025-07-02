
import numpy as np
import numpy.typing as npt
from nes.dists import Dist, DiagNormal, XDiagNormal
from nes.models import NPolynome
from nes.optimizer import Optimizer
from nes.models.benchmarks2D import Ackley
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
        error[i] = np.linalg.norm(optimizer.dist.get_optimum() )#- model.optimum)
    return error


def plot_dims_analysis(dims, Distributions, sample_num = 100, n_steps = 1000, initial_guess=None):


    errors = np.zeros((len(Distributions), len(dims)))

    for i, Distribution in enumerate(Distributions):
        for j, dim in enumerate(dims):
            weights = np.array([[0] * dim, [0] * dim, [1.] * dim])
            model = NPolynome(weights)
            model.lower = np.full(weights.shape[1], -20)
            model.upper = np.full(weights.shape[1], +20)

            dist = Distribution.from_model(model)

            if initial_guess is not None:
                if len(initial_guess) != len(dims):
                    raise ValueError("List of initial guesses does not match dimensions")
                dist.set_init_params(initial_guess)

            optimizer = Optimizer(model, dist, sample_num)

            for _ in range(n_steps):
                optimizer.step()

            error = np.linalg.norm(dist.get_optimum())
            errors[i, j] = error
            # print(f"{Distribution.__name__} | dim: {dim} | error: {error}")

    for i, Distribution in enumerate(Distributions):
        plt.loglog(dims, errors[i], label=Distribution.__name__)

    plt.xlabel("Dimension")
    plt.ylabel(r"$L_2 $ Error")
    plt.title(f"Optimization Error vs Dimension for {n_steps} steps and {sample_num} samples ")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_samples_analysis(sample_numbers, Distributions, Model, max_steps=1000, initial_guess=None):
    """Expects a Model from benchmarks2D"""
    steps = np.ones((len(Distributions), len(sample_numbers)))*max_steps


    for i, Distribution in enumerate(Distributions):
        for j, sample_number in enumerate(sample_numbers):
            model = Model.model()
            dist = Distribution.from_model(model)
            if initial_guess is not None:
                dist.set_init_params(initial_guess)
            optimizer = Optimizer(model, dist,sample_number)
            for n in range(max_steps):
                optimizer.step()
                if np.linalg.norm(dist.get_optimum()) < 1e-3:
                    steps[i, j] = n
                    break
    for i, Distribution in enumerate(Distributions):
        plt.loglog(sample_numbers, steps[i], label=Distribution.__name__)

    plt.xlabel("Dimension")
    plt.ylabel(r"$L_2 $ Error")
    plt.title("Samples ")
    plt.legend()
    plt.grid(True)
    plt.show()





dims = np.logspace(1, 8, 6, base=2, dtype=int)

plot_samples_analysis(dims, [DiagNormal, XDiagNormal], Ackley)

print(dims)
# plot_dims_analysis(dims, [DiagNormal, XDiagNormal])



# my_model = Ackley.model()
# my_conv = convergence(my_model, XDiagNormal.from_model(my_model), 100)
# plt.plot(my_conv)
# plt.show()
