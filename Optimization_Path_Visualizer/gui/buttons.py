import tkinter as tk

from utils.dimension_detector import detect_dimension

from visualization.animation_1d import animate_1d
from visualization.animation_3d import animate_3d
from visualization.compare_optimizers import compare_optimizers

from algorithms.gradient_descent import gradient_descent_1d, gradient_descent_2d
from algorithms.sgd import sgd_1d, sgd_2d
from algorithms.momentum import momentum_1d, momentum_2d
from algorithms.rmsprop import rmsprop_1d, rmsprop_2d
from algorithms.adam import adam_1d, adam_2d

# GD ======================================================
def run_gd(inputs):

    func = inputs["func"].get()
    derivative = inputs["derivative"].get()
    lr = float(inputs["lr"].get())
    iterations = int(inputs["iterations"].get())

    dimension = detect_dimension(func)

    if dimension == 1:

        x0 = float(inputs["x0"].get())

        xs = gradient_descent_1d(derivative, x0, lr, iterations)

        animate_1d(func, xs)

    else:

        dx_str, dy_str = derivative.split(";")

        x0 = float(inputs["x0"].get())
        y0 = float(inputs["y0"].get())

        xs, ys = gradient_descent_2d(dx_str, dy_str, x0, y0, lr, iterations)

        animate_3d(func, xs, ys)

# SGD =====================================================
def run_sgd(inputs):

    func = inputs["func"].get()
    derivative = inputs["derivative"].get()
    lr = float(inputs["lr"].get())
    iterations = int(inputs["iterations"].get())

    dimension = detect_dimension(func)

    if dimension == 1:

        x0 = float(inputs["x0"].get())

        xs = sgd_1d(derivative, x0, lr, iterations)

        animate_1d(func, xs)

    else:

        dx_str, dy_str = derivative.split(";")

        x0 = float(inputs["x0"].get())
        y0 = float(inputs["y0"].get())

        xs, ys = sgd_2d(dx_str, dy_str, x0, y0, lr, iterations)

        animate_3d(func, xs, ys)

# Momentum ================================================
def run_momentum(inputs):

    func = inputs["func"].get()
    derivative = inputs["derivative"].get()
    lr = float(inputs["lr"].get())
    iterations = int(inputs["iterations"].get())

    dimension = detect_dimension(func)

    if dimension == 1:

        x0 = float(inputs["x0"].get())

        xs = momentum_1d(derivative, x0, lr, iterations)

        animate_1d(func, xs)

    else:

        dx_str, dy_str = derivative.split(";")

        x0 = float(inputs["x0"].get())
        y0 = float(inputs["y0"].get())

        xs, ys = momentum_2d(dx_str, dy_str, x0, y0, lr, iterations)

        animate_3d(func, xs, ys)

# RmsProp =================================================
def run_rmsprop(inputs):

    func = inputs["func"].get()
    derivative = inputs["derivative"].get()
    lr = float(inputs["lr"].get())
    iterations = int(inputs["iterations"].get())

    dimension = detect_dimension(func)

    if dimension == 1:

        x0 = float(inputs["x0"].get())

        xs = rmsprop_1d(derivative, x0, lr, iterations)

        animate_1d(func, xs)

    else:

        dx_str, dy_str = derivative.split(";")

        x0 = float(inputs["x0"].get())
        y0 = float(inputs["y0"].get())

        xs, ys = rmsprop_2d(dx_str, dy_str, x0, y0, lr, iterations)

        animate_3d(func, xs, ys)

# Adam ====================================================
def run_adam(inputs):

    func = inputs["func"].get()
    derivative = inputs["derivative"].get()
    lr = float(inputs["lr"].get())
    iterations = int(inputs["iterations"].get())

    dimension = detect_dimension(func)

    if dimension == 1:

        x0 = float(inputs["x0"].get())

        xs = adam_1d(derivative, x0, lr, iterations)

        animate_1d(func, xs)

    else:

        dx_str, dy_str = derivative.split(";")

        x0 = float(inputs["x0"].get())
        y0 = float(inputs["y0"].get())

        xs, ys = adam_2d(dx_str, dy_str, x0, y0, lr, iterations)

        animate_3d(func, xs, ys)

# Comparison ==============================================
def run_comparison(inputs):

    func = inputs["func"].get()
    derivative = inputs["derivative"].get()

    lr = float(inputs["lr"].get())
    iterations = int(inputs["iterations"].get())

    dimension = detect_dimension(func)

    if dimension == 1:

        x0 = float(inputs["x0"].get())

        gd = gradient_descent_1d(derivative, x0, lr, iterations)
        sgd = sgd_1d(derivative, x0, lr, iterations)
        momentum = momentum_1d(derivative, x0, lr, iterations)
        rmsprop = rmsprop_1d(derivative, x0, lr, iterations)
        adam = adam_1d(derivative, x0, lr, iterations)

        paths = {
            "GD": gd,
            "SGD": sgd,
            "Momentum": momentum,
            "RMSProp": rmsprop,
            "Adam": adam
        }

        compare_optimizers(func, paths)

    else:

        dx_str, dy_str = derivative.split(";")

        x0 = float(inputs["x0"].get())
        y0 = float(inputs["y0"].get())

        gd = gradient_descent_2d(dx_str, dy_str, x0, y0, lr, iterations)
        sgd = sgd_2d(dx_str, dy_str, x0, y0, lr, iterations)
        momentum = momentum_2d(dx_str, dy_str, x0, y0, lr, iterations)
        rmsprop = rmsprop_2d(dx_str, dy_str, x0, y0, lr, iterations)
        adam = adam_2d(dx_str, dy_str, x0, y0, lr, iterations)

        paths = {
            "GD": gd,
            "SGD": sgd,
            "Momentum": momentum,
            "RMSProp": rmsprop,
            "Adam": adam
        }

        compare_optimizers(func, paths)

def create_buttons(root, inputs):

    frame = tk.Frame(root)
    frame.pack(pady=20)

    btn_gd = tk.Button(frame, text="Gradient Descent", width=20, command=lambda: run_gd(inputs))
    btn_gd.grid(row=0, column=0, padx=10, pady=10)

    btn_sgd = tk.Button(frame, text="SGD", width=20, command=lambda: run_sgd(inputs))
    btn_sgd.grid(row=0, column=1, padx=10, pady=10)

    btn_momentum = tk.Button(frame, text="Momentum", width=20, command=lambda: run_momentum(inputs))
    btn_momentum.grid(row=1, column=0, padx=10, pady=10)

    btn_rmsprop = tk.Button(frame, text="RMSProp", width=20, command=lambda: run_rmsprop(inputs))
    btn_rmsprop.grid(row=1, column=1, padx=10, pady=10)

    btn_adam = tk.Button(frame, text="Adam", width=20, command=lambda: run_adam(inputs))
    btn_adam.grid(row=2, column=0, padx=10, pady=10)

    btn_compare = tk.Button(frame, text="Compare Optimizers", width=20, command=lambda: run_comparison(inputs))
    btn_compare.grid(row=2, column=1, padx=10, pady=10)