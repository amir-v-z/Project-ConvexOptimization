import math

from utils.math_functions import evaluate_derivative

def adam_1d(derivative_str, x0, lr, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):

    xs = [x0]
    x = x0

    m = 0
    v = 0

    for t in range(1, iterations + 1):

        grad = evaluate_derivative(x, derivative_str)
        if grad is None:
            break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x = x - lr * m_hat / (math.sqrt(v_hat) + epsilon)

        xs.append(x)

    return xs

def adam_2d(dx_str, dy_str, x0, y0, lr, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):

    xs = [x0]
    ys = [y0]

    x = x0
    y = y0

    mx = 0
    my = 0
    vx = 0
    vy = 0

    for t in range(1, iterations + 1):

        grad_x = evaluate_derivative((x, y), dx_str)
        grad_y = evaluate_derivative((x, y), dy_str)

        if grad_x is None or grad_y is None:
            break

        mx = beta1 * mx + (1 - beta1) * grad_x
        my = beta1 * my + (1 - beta1) * grad_y

        vx = beta2 * vx + (1 - beta2) * (grad_x ** 2)
        vy = beta2 * vy + (1 - beta2) * (grad_y ** 2)

        mx_hat = mx / (1 - beta1 ** t)
        my_hat = my / (1 - beta1 ** t)

        vx_hat = vx / (1 - beta2 ** t)
        vy_hat = vy / (1 - beta2 ** t)

        x = x - lr * mx_hat / (math.sqrt(vx_hat) + epsilon)
        y = y - lr * my_hat / (math.sqrt(vy_hat) + epsilon)

        xs.append(x)
        ys.append(y)

    return xs, ys