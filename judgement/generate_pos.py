import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv
import pybullet as p
import numpy as np
env = ContinuumRobotEnv(render_mode=False, control_mode=1,connection_mode=p.DIRECT)
reachable_points = []

# 可调节的参数
num_samples = 10000

for _ in range(num_samples):
    action = env.action_space.sample()  # 随机动作
    env.sim.set_robot(*action)
    pos = env.sim.get_end_effector_position()
    reachable_points.append(pos)

# 保存为 numpy 文件
np.save("reachable_targets_medium.npy", np.array(reachable_points))
print(f"✅ 成功保存 {len(reachable_points)} 个可达目标点")
