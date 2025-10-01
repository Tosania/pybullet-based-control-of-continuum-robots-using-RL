import numpy as np
import matplotlib.pyplot as plt

# 定义 x 区间和 y 值
x = np.linspace(0 ,2, 100000)  # 生成1000个点
def f(distance):
    reward = -distance + np.exp(-120 * distance**2) * 5
    return reward

y = f(x)  # 目标函数
 #reward = -distance**2 + np.exp(-distance**2) * 10
#reward = -distance + 10 * (1 / (1 + distance))

# 绘图
plt.figure(figsize=(8, 4))       # 设置图像大小
plt.plot(x, y, label='y = sin(x)', linewidth=2)  # 画曲线
plt.title("Function Curve: y = sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
