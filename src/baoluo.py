import numpy as np
import matplotlib.pyplot as plt

# 读取文件
targets = np.load("./source/reachable_targets.npy")  # 请确保路径正确，或使用绝对路径

# 提取 x 和 y^2 + z^2
x_values = targets[:, 0]
y2_z2_values = targets[:, 1]**2 + targets[:, 2]**2

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y2_z2_values, alpha=0.6)
plt.xlabel('x')
plt.ylabel('y² + z²')
plt.title('2D Plot of x vs y² + z²')
plt.grid(True)
plt.show()
