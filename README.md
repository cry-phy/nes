# NES

## Introduction

Natural Evolution Strategies (NES) are a family of evolutionary
optimization methods that search a (possibly high-dimensional) fitness
landscape by adapting the parameters of a sampling distribution instead
of a single point. Each iteration draws samples, evaluates their fitness,
and estimates a stochastic gradient via the log-likelihood trick. The
plain gradient becomes unstable as the variance shrinks, so NES
precondition it with the inverse Fisher information matrix. The result is
the *natural gradient*, which is invariant to the parametrization and
stays stable close to the optimum.

This project implements several NES variants so they can be compared on
stability and performance:

- Separable multivariate normal (plain and exponential parametrization, SNES)
- Full multivariate normal (plain and exponential parametrization, xNES)
- Heavy-tailed Student's t (exponential parametrization), which helps to
  escape local minima

Two learning-rate strategies are available: constant heuristic rates
([Schaul et al. 2011](https://www.idsia.ch/~juergen/gecco2011snes.pdf))
and adaptive KL-budget rates with EMA stabilization. The code consists of
**Models** (objective functions such as Sphere, Rastrigin, Ackley,
Rosenbrock and Eggholder), **Distributions** and **Optimizers**.

![All models](res/allmodels.png)

The full theoretical background, including the derivation of the
gradients and Fisher matrices, is in the [Handout](res/Handout.md).

## Results

**Convergence.** On the Ackley function, the diagonal normal with
exponential parametrization finds the global minimum with essentially
exponential convergence, despite initial overshoots and short stagnant
phases.

![Ackley convergence, XDiagNormalAdaptive](res/Ackley_Conv.png)

The animations show how the distribution adapts to the landscape:

![Rastrigin xNES](res/xnes_rastrigin.gif)

**Dimensional performance.** Ensemble runs on the sphere function show
exponential convergence in 2D, where the learning-rate strategy makes no
notable difference. In 128D, the KL-budget rates clearly outperform the
heuristic rates.

| 2D Sphere | 128D Sphere |
|---|---|
| ![2D Sphere](res/lowDSphere.png) | ![128D Sphere](res/highDSphere.png) |

**Model-distribution compatibility.** Multistart ensembles over every
model and distribution give the averaged Euclidean distance to the
optimum after termination:

![Compatibility matrix](res/capabilities_multistart.png)

Some combinations clearly outperform others, but no distribution wins on
every function, as expected from the No Free Lunch theorems. Premature
convergence into local minima, visible for Rastrigin and Ackley, remains
the main weakness.

## Summary

Even this barebones implementation handles a wide variety of problems
adequately. KL-based adaptive learning rates are the better choice for
high-dimensional problems. The remaining issue of local minima could be
addressed with restarts, heat-based variance expansion or more advanced
learning-rate schemes.

## Usage

 Look at [Examples](res/example.ipynb)!

This project uses poetry for managing dependencies for Python 3.10. To use use the environment run
```bash
poetry install --no-root
poetry env activate
```

To install pre-commit hooks use
```bash
poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## License

Copyright (C) 2025 Christof Stein <christof.stein@student.uibk.ac.at>  
Copyright (C) 2025 Felix Jannik Möller <felix.moeller@student.uibk.ac.at>

This project is licensed under the GNU Lesser General Public License v3.0.
See [LICENSE](LICENSE) for the full text.
