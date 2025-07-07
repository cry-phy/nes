import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from nes.dists import Dist, DiagNormal, XDiagNormal
from nes.models import NPolynome
from nes.optimizer import Optimizer
from nes.models.benchmarks2D import Ackley


def convergence(
    model, dist: Dist, sample_num: int, n_steps=1000, initial_guess=None
) -> npt.NDArray[np.float64]:
    """Run NES optimization and return L2 error over time."""
    if initial_guess is not None:
        dist.set_init_params(initial_guess)

    error = np.zeros(n_steps)
    optimizer = Optimizer(model, dist, sample_num)
    for i in range(n_steps):
        optimizer.step()
        error[i] = np.linalg.norm(optimizer.dist.get_optimum())
    return error


def plot_dims_analysis(
    dims, distribtutions, sample_num=100, n_steps=1000, initial_guess=None
):
    """Plot error vs dimension for given distributions."""
    errors = np.zeros((len(distribtutions), len(dims)))

    for i, distribtution in enumerate(distribtutions):
        for j, dim in enumerate(dims):
            weights = np.array([[0] * dim, [0] * dim, [1.0] * dim])
            model = NPolynome(weights)
            model.lower = np.full(dim, -20)
            model.upper = np.full(dim, +20)

            dist = distribtution.from_model(model)
            if initial_guess is not None:
                dist.set_init_params(initial_guess[j])

            optimizer = Optimizer(model, dist, sample_num)
            for _ in range(n_steps):
                optimizer.step()

            errors[i, j] = np.linalg.norm(dist.get_optimum())

    for i, distribtution in enumerate(distribtutions):
        plt.loglog(dims, errors[i], label=distribtution.__name__)

    plt.xlabel("Dimension")
    plt.ylabel(r"$L_2$ Error")
    plt.title(
        f"Optimization Error vs Dimension ({n_steps} steps, {sample_num} samples)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_samples_analysis(
    sample_numbers, distribtutions, model, max_steps=1000, initial_guess=None
):
    """Plot number of steps to convergence for varying sample numbers."""
    steps = np.ones((len(distribtutions), len(sample_numbers))) * max_steps

    for i, distribtution in enumerate(distribtutions):
        for j, sample_number in enumerate(sample_numbers):
            model = model.model()
            dist = distribtution.from_model(model)
            if initial_guess is not None:
                dist.set_init_params(initial_guess)

            optimizer = Optimizer(model, dist, sample_number)
            for n in range(max_steps):
                optimizer.step()
                if np.linalg.norm(dist.get_optimum() - model.optimum) < 1e-3:
                    steps[i, j] = n
                    break

    for i, distribtution in enumerate(distribtutions):
        plt.loglog(sample_numbers, steps[i], label=distribtution.__name__)

    plt.xlabel("Samples")
    plt.ylabel("Steps to Convergence")
    plt.title("Sample Efficiency Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_minima_analysis(
    num_runs: int,
    distribtutions,
    Model,
    sample_number=100,
    steps=1000,
    initial_guess=None,
):
    """Plot distance to global optimum across multiple runs."""
    distances = np.zeros((len(distribtutions), num_runs))

    for i, distribtution in enumerate(distribtutions):
        for j in range(num_runs):
            model = Model.model()
            dist = distribtution.from_model(model)
            if initial_guess is not None:
                dist.set_init_params(initial_guess)

            optimizer = Optimizer(model, dist, sample_number)
            for _ in range(steps):
                optimizer.step()

            distances[i, j] = np.linalg.norm(dist.get_optimum() - model.optimum)

    for i, distribtution in enumerate(distribtutions):
        plt.plot(
            np.arange(num_runs), distances[i], marker="o", label=distribtution.__name__
        )

    plt.xlabel("Run")
    plt.ylabel("Distance from Global Minimum")
    plt.title("Benchmarking Local Minima")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # dims = np.logspace(1, 8, 6, base=2, dtype=int)
    # plot_dims_analysis(dims, [DiagNormal, XDiagNormal])
    # plot_samples_analysis(dims, [DiagNormal, XDiagNormal], Ackley)
    plot_minima_analysis(30, [DiagNormal, XDiagNormal], Ackley)

    # For single convergence curve:
    # model = Ackley.model()
    # dist = XDiagNormal.from_model(model)
    # conv = convergence(model, dist, 100)
    # plt.plot(conv)
    # plt.show()
