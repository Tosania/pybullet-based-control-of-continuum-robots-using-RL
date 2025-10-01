import numpy as np
import matplotlib.pyplot as plt
import random

# 采样函数
def sample_point_in_sphere(radius=2.5, center=[0, 0, 0]):
    x = random.uniform(-4.5, 4.5)
    r = np.sqrt(25 - 25.0 / 4.5 * abs(x)) * random.uniform(0, 1)
    coss = random.uniform(0, 1)
    sins = np.sqrt(1 - coss * coss)
    return [x, r * coss, r * sins]

# 生成若干个点
num_points = 30000
points = np.array([sample_point_in_sphere() for _ in range(num_points)])

# 保存为 .npy 文件

# 可视化
x_vals = points[:, 0]
y2_z2_vals = points[:, 1]**2 + points[:, 2]**2

plt.figure(figsize=(8, 6))
plt.scatter(x_vals, y2_z2_vals, alpha=0.6)
plt.xlabel('x')
plt.ylabel('y² + z²')
plt.title('Scatter Plot of Sampled Points: x vs y² + z²')
plt.grid(True)
plt.show()
