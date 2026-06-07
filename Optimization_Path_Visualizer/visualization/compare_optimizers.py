import numpy as np
import matplotlib.pyplot as plt

from utils.math_functions import evaluate_function


def compare_optimizers(func_str, paths):

    first_path = list(paths.values())[0]

    colors = {
        "GD": "red",
        "SGD": "orange",
        "Momentum": "blue",
        "RMSProp": "green",
        "Adam": "purple"
    }

    # ================= 1D =================
    if isinstance(first_path, list):

        x_vals = np.linspace(-2, 5, 400)
        y_vals = [evaluate_function(x, func_str) for x in x_vals]

        plt.figure(figsize=(8,6))

        plt.plot(x_vals, y_vals, color="black", label="f(x)")

        for name, xs in paths.items():

            ys = [evaluate_function(x, func_str) for x in xs]

            plt.plot(xs, ys, marker="o", color=colors[name], label=name)

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Optimizer Comparison")
        plt.grid(True)
        plt.legend()

        plt.show()

    # ================= 2D =================
    else:

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")

        x = np.linspace(-5,5,100)
        y = np.linspace(-5,5,100)
        X, Y = np.meshgrid(x,y)

        Z = evaluate_function((X,Y), func_str)

        ax.plot_surface(X, Y, Z, alpha=0.5, cmap="viridis")

        for name, path in paths.items():

            xs, ys = path
            zs = [evaluate_function((x,y), func_str) for x,y in zip(xs,ys)]

            ax.plot(xs, ys, zs, marker="o", color=colors[name], label=name)

        ax.set_title("Optimizer Comparison (3D)")
        ax.legend()

        plt.show()