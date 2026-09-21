import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# خواندن داده
data = pd.read_csv("./Project-ConvexOptimization/SGD-MBGD/dataset.csv")

# گرفتن پارامترها از کاربر
learning_rate = float(input("Enter Learning Rate: "))
epochs = int(input("Enter number of epochs: "))
batch_size = int(input("Enter mini-batch size: "))

# استخراج ستون‌های داده
X = data["x"].values.reshape(-1,1)
y = data["y"].values.reshape(-1,1)

# (bias) اضافه کردن بایاس
X_b = np.c_[np.ones((len(X),1)), X]

# تابع هزینه
def compute_cost(theta, X, y):
    m = len(y)
    return (1/(2*m)) * np.sum((X.dot(theta) - y)**2) # MSE

# -------------------------
# SGD
# -------------------------
def SGD(X, y, lr, epochs):

    m = len(y)
    theta = np.random.randn(2,1)

    path = []

    start = time.time()
    iterations = 0

    for epoch in range(epochs):

        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):

            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]

            gradient = xi.T.dot(xi.dot(theta) - yi)

            theta = theta - lr * gradient

            path.append(theta.copy())

            iterations += 1

    runtime = time.time() - start

    return theta, iterations, runtime, np.array(path)


# -------------------------
# Mini Batch
# -------------------------
def MiniBatchGD(X, y, lr, epochs, batch_size):

    m = len(y)
    theta = np.random.randn(2,1)

    path = []

    start = time.time()
    iterations = 0

    for epoch in range(epochs):

        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):

            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            gradient = (1/len(xi)) * xi.T.dot(X_shuffled[i:i+batch_size].dot(theta) - yi)

            theta = theta - lr * gradient

            path.append(theta.copy())

            iterations += 1

    runtime = time.time() - start

    return theta, iterations, runtime, np.array(path)


# اجرای الگوریتم‌ها
theta_sgd, iter_sgd, time_sgd, path_sgd = SGD(X_b, y, learning_rate, epochs)
theta_mb, iter_mb, time_mb, path_mb = MiniBatchGD(X_b, y, learning_rate, epochs, batch_size)

print()
print("SGD \nTime:", time_sgd, "\nIterations:", iter_sgd)
print("-"*10)
print("MiniBatch \nTime:", time_mb, "\nIterations:", iter_mb)


# -------------------------
# رسم کانتور تابع هزینه
# -------------------------

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-10, 10, 100)

J_vals = np.zeros((100,100))

for i in range(100):
    for j in range(100):
        t = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[i,j] = compute_cost(t, X_b, y)

theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)


plt.figure(figsize=(8,6))

plt.contour(theta0_grid, theta1_grid, J_vals.T, levels=30)

# SGD مسیر
plt.plot(path_sgd[:,0], path_sgd[:,1], 'r-', label="SGD Path", alpha=0.6)

# MiniBatch مسیر
plt.plot(path_mb[:,0], path_mb[:,1], 'b-', label="MiniBatch Path", alpha=0.8)

plt.xlabel("theta0")
plt.ylabel("theta1")

plt.title("Optimization Path (SGD vs MiniBatch)")
plt.legend()

plt.show()