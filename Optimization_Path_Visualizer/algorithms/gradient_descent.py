from utils.math_functions import evaluate_derivative

def gradient_descent_1d(derivative_str, x0, lr, iterations):

    xs = [x0]
    x = x0

    for _ in range(iterations):

        grad = evaluate_derivative(x, derivative_str)
        if grad is None:
            break

        x = x - lr * grad
        xs.append(x)

    return xs

def gradient_descent_2d(dx_str, dy_str, x0, y0, lr, iterations):

    xs = [x0]
    ys = [y0]

    x = x0
    y = y0

    for _ in range(iterations):

        grad_x = evaluate_derivative((x, y), dx_str)
        grad_y = evaluate_derivative((x, y), dy_str)

        if grad_x is None or grad_y is None:
            break

        x = x - lr * grad_x
        y = y - lr * grad_y

        xs.append(x)
        ys.append(y)

    return xs, ys