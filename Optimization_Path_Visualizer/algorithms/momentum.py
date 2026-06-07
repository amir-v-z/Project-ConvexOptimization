from utils.math_functions import evaluate_derivative

def momentum_1d(derivative_str, x0, lr, iterations, beta=0.9):

    xs = [x0]
    x = x0
    v = 0

    for _ in range(iterations):

        grad = evaluate_derivative(x, derivative_str)
        if grad is None:
            break

        v = beta * v + (1 - beta) * grad
        x = x - lr * v

        xs.append(x)

    return xs

def momentum_2d(dx_str, dy_str, x0, y0, lr, iterations, beta=0.9):

    xs = [x0]
    ys = [y0]

    x = x0
    y = y0

    vx = 0
    vy = 0

    for _ in range(iterations):

        grad_x = evaluate_derivative((x, y), dx_str)
        grad_y = evaluate_derivative((x, y), dy_str)

        if grad_x is None or grad_y is None:
            break

        vx = beta * vx + (1 - beta) * grad_x
        vy = beta * vy + (1 - beta) * grad_y

        x = x - lr * vx
        y = y - lr * vy

        xs.append(x)
        ys.append(y)

    return xs, ys