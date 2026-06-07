import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from utils.math_functions import evaluate_function

def animate_3d(func_str, xs, ys):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(min(xs)-2, max(xs)+2, 100)
    Y = np.linspace(min(ys)-2, max(ys)+2, 100)

    X, Y = np.meshgrid(X, Y)

    Z = evaluate_function((X, Y), func_str)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

    zs = [evaluate_function((x, y), func_str) for x, y in zip(xs, ys)]

    point, = ax.plot([], [], [], "ro", markersize=8)
    path, = ax.plot([], [], [], "r--")

    path_x = []
    path_y = []
    path_z = []

    def update(frame):

        x = xs[frame]
        y = ys[frame]
        z = zs[frame]

        path_x.append(x)
        path_y.append(y)
        path_z.append(z)

        point.set_data([x], [y])
        point.set_3d_properties([z])

        path.set_data(path_x, path_y)
        path.set_3d_properties(path_z)

        return point, path

    ani = FuncAnimation(
        fig,
        update,
        frames=len(xs),
        interval=500,
        repeat=False
    )

    ax.set_title("3D Animation")

    plt.show()