import numpy as np
import pandas as pd

# برای تکرارپذیری
np.random.seed(42)

# تولید 1000 داده
n = 1000
X = 2 * np.random.rand(n, 1)
y = 4 + 3 * X + np.random.randn(n, 1)

# ساخت دیتافریم
data = pd.DataFrame({
    "x": X.flatten(),
    "y": y.flatten()
})

# ذخیره در فایل
data.to_csv("./Project-ConvexOptimization/SGD-MBGD/dataset.csv", index=False)

print("Dataset with 1000 samples saved to dataset.csv")