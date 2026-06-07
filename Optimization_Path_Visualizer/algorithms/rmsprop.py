import math

from utils.math_functions import evaluate_derivative

def rmsprop_1d(derivative_str, x0, lr, iterations, beta=0.9, epsilon=1e-8):

    xs = [x0]
    x = x0
    s = 0

    for _ in range(iterations):

        grad = evaluate_derivative(x, derivative_str)
        if grad is None:
            break

        s = beta * s + (1 - beta) * (grad ** 2)

        x = x - lr * grad / (math.sqrt(s) + epsilon)

        xs.append(x)

    return xs

def rmsprop_2d(dx_str, dy_str, x0, y0, lr, iterations, beta=0.9, epsilon=1e-8):

    xs = [x0]
    ys = [y0]

    x = x0
    y = y0

    sx = 0
    sy = 0

    for _ in range(iterations):

        grad_x = evaluate_derivative((x, y), dx_str)
        grad_y = evaluate_derivative((x, y), dy_str)

        if grad_x is None or grad_y is None:
            break

        sx = beta * sx + (1 - beta) * (grad_x ** 2)
        sy = beta * sy + (1 - beta) * (grad_y ** 2)

        x = x - lr * grad_x / (math.sqrt(sx) + epsilon)
        y = y - lr * grad_y / (math.sqrt(sy) + epsilon)

        xs.append(x)
        ys.append(y)

    return xs, ys