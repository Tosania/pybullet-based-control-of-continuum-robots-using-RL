import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 引入 3D 坐标系支持

# 加载点库
points = np.load("reachable_targets.npy")

# 计算每个点到原点的距离
distances = np.linalg.norm(points, axis=1)

# 设置距离限制
min_distance = 2.4 # 最小距离
max_distance = 2.5  # 最大距离

# 筛选满足距离条件的点
mask = (distances >= min_distance) & (distances <= max_distance)
filtered_points = points[mask]

# 拆分 x, y, z 坐标
x = filtered_points[:, 0]
y = filtered_points[:, 1]
z = filtered_points[:, 2]

# 创建 3D 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制筛选后的点
ax.scatter(x, y, z, c='blue', s=10, alpha=0.6)

# 绘制距离限制球面
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# 绘制最小距离球面
x_min = min_distance * np.outer(np.cos(u), np.sin(v))
y_min = min_distance * np.outer(np.sin(u), np.sin(v))
z_min = min_distance * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_min, y_min, z_min, color='r', alpha=0.1)

# 绘制最大距离球面
x_max = max_distance * np.outer(np.cos(u), np.sin(v))
y_max = max_distance * np.outer(np.sin(u), np.sin(v))
z_max = max_distance * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_max, y_max, z_max, color='g', alpha=0.1)

# 设置坐标轴范围
max_range = 3
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# 添加标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'筛选后的目标点分布 (距离范围: {min_distance}-{max_distance})')

# 保存筛选后的点
np.save("filtered_targets.npy", filtered_points)

plt.show()
