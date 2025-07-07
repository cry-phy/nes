import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nes.models.benchmarks2D import *
from nes.dists import *

# Setup
model = BukinN6.model()
model.upper[:] = 15
model.lower[:] = -10

fig, ax = plt.subplots()
dist = None  # will be set in init()

def init():
    """Reset distribution and draw static fitness background."""
    global dist
    dist = XNormalAdaptive.from_model(model)
    dist._param_mean[0] = 3
    dist._param_mean[1] = 6

    ax.clear()
    model.plot_fitness(ax, model.lower, model.upper)
    return []

def update(frame):
    """Update distribution and redraw sigma contours."""
    ax.clear()
    model.plot_fitness(ax, model.lower, model.upper)
    n = 1
    for _ in range(n):  # steps per frame
        samples = dist.get_samples(64)
        fitness = model.evaluate_samples(samples)
        dist.param_variation(samples, fitness)

    dist.plot_sigmas(ax)
    ax.set_title(f"Iteration: {frame * n}")
    return []

ani = FuncAnimation(
    fig,
    update,
    frames=150,
    init_func=init,
    blit=False,
    interval=10,
    repeat=True
)

ani.save("xnes_rastrigin.gif",writer=PillowWriter(fps=60),dpi=150,)

plt.show()
