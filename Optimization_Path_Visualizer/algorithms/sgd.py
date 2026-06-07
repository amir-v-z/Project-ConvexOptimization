import random

from utils.math_functions import evaluate_derivative

def sgd_1d(derivative_str, x0, lr, iterations, noise_scale=0.1):

    xs = [x0]
    x = x0

    for _ in range(iterations):

        grad = evaluate_derivative(x, derivative_str)
        if grad is None:
            break

        noise = random.uniform(-noise_scale, noise_scale)
        grad = grad + noise

        x = x - lr * grad
        xs.append(x)

    return xs

def sgd_2d(dx_str, dy_str, x0, y0, lr, iterations, noise_scale=0.1):

    xs = [x0]
    ys = [y0]

    x = x0
    y = y0

    for _ in range(iterations):

        grad_x = evaluate_derivative((x, y), dx_str)
        grad_y = evaluate_derivative((x, y), dy_str)

        if grad_x is None or grad_y is None:
            break

        noise_x = random.uniform(-noise_scale, noise_scale)
        noise_y = random.uniform(-noise_scale, noise_scale)

        grad_x = grad_x + noise_x
        grad_y = grad_y + noise_y

        x = x - lr * grad_x
        y = y - lr * grad_y

        xs.append(x)
        ys.append(y)

    return xs, ys