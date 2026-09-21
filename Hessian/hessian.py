import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def newton_method(f_grad, f_hessian, x0, tol=1e-6, max_iter=10, verbose=True):
    x = x0.astype(float)
    path = [x.copy()]

    if verbose:
        print(f"Starting at: {x}")

    for i in range(max_iter):
        grad = f_grad(x)
        hess = f_hessian(x)

        step = np.linalg.solve(hess, grad)
        x_new = x - step
        path.append(x_new.copy())

        if verbose:
            print(f"Iteration {i+1}: x = {x_new}")

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return x, np.array(path)

def f_2d(x, y):
    return x**2 + 2*y**2

def grad_2d(x_vec):
    x, y = x_vec
    return np.array([2*x, 4*y])

def hessian_2d(x_vec):
    return np.array([[2, 0],
                     [0, 4]])

print("\n--- Testing 2D Function ---")

x0_2d = np.array([10.0, -5.0])
_, path_2d = newton_method(grad_2d, hessian_2d, x0_2d)

x = np.linspace(-11, 11, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = f_2d(X, Y)

plt.figure(figsize=(8,6))
plt.contour(X, Y, Z, levels=30)
plt.plot(path_2d[:,0], path_2d[:,1], 'ro-', label="Newton Path")
plt.scatter(0, 0, c='green', s=100, label="Minimum")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Newton Method Path (2D): x^2 + 2y^2")
plt.legend()
plt.grid(True)
plt.show()

def f_3d(x, y, z):
    return x**2 + y**2 + z**2 + x*y

def grad_3d(x_vec):
    x, y, z = x_vec
    return np.array([2*x + y, 2*y + x, 2*z])

def hessian_3d(x_vec):
    return np.array([[2, 1, 0],
                     [1, 2, 0],
                     [0, 0, 2]])

print("\n--- Testing 3D Function ---")

x0_3d = np.array([1.0, 2.0, 3.0])
_, path_3d = newton_method(grad_3d, hessian_3d, x0_3d)

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = f_3d(X, Y, 0)

ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

ax.plot(path_3d[:,0], path_3d[:,1], path_3d[:,2], 'ro-', linewidth=2, label="Newton Path")

ax.scatter(0, 0, 0, c='green', s=100, label="Minimum")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Newton Method Path (3D): x^2 + y^2 + z^2 + xy")
ax.legend()
plt.show()

def f_2d_example3(x, y):
    return x**4 + y**4

def grad_2d_example3(x_vec):
    x, y = x_vec
    return np.array([4*x**3, 4*y**3])

def hessian_2d_example3(x_vec):
    x, y = x_vec
    return np.array([[12*x**2, 0],
                     [0, 12*y**2]])

print("\n--- Testing 2D Function (Non-Quadratic 2D) ---")

x0_3rd = np.array([2.0, 1.5])
_, path_3rd = newton_method(grad_2d_example3, hessian_2d_example3, x0_3rd, max_iter=10)

x = np.linspace(-2.5, 2.5, 400)
y = np.linspace(-2.5, 2.5, 400)
X, Y = np.meshgrid(x, y)
Z = f_2d_example3(X, Y)

plt.figure(figsize=(8,6))
plt.contour(X, Y, Z, levels=30)
plt.plot(path_3rd[:,0], path_3rd[:,1], 'bo-', label="Newton Path")
plt.scatter(0, 0, c='green', s=100, label="Minimum")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Newton Method Path (2D): x^4 + y^4")
plt.legend()
plt.grid(True)
plt.show()