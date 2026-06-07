import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from utils.math_functions import evaluate_function

def animate_1d(func_str, xs):

    ys = [evaluate_function(x, func_str) for x in xs]

    X = np.linspace(min(xs)-2, max(xs)+2, 400)
    Y = [evaluate_function(x, func_str) for x in X]

    fig, ax = plt.subplots()

    ax.plot(X, Y, label="f(x)")

    point, = ax.plot([], [], "ro")

    path_x = []
    path_y = []

    line, = ax.plot([], [], "r--", alpha=0.6)

    def update(frame):

        x = xs[frame]
        y = ys[frame]

        path_x.append(x)
        path_y.append(y)

        point.set_data([x], [y])
        line.set_data(path_x, path_y)

        return point, line

    ani = FuncAnimation(
        fig,
        update,
        frames=len(xs),
        interval=500,
        repeat=False
    )

    plt.title("Animation")
    plt.legend()

    plt.show()