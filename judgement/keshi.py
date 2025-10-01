import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 引入 3D 坐标系支持

# 加载点库
points = np.load("reachable_targets.npy")

# 拆分 x, y, z 坐标
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', s=10, alpha=0.6)

ax.set_title("Reachable End-Effector Positions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
