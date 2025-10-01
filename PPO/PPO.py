from stable_baselines3 import PPO
import sys
import os
import pybullet as p
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv
import time

# 加载训练好的模型
model_path = "continuum_robot_ppo"
model = PPO.load(model_path)

# 创建环境
env = ContinuumRobotEnv(control_mode=0,connection_mode=p.GUI)

# 重置环境
obs, _ = env.reset()

# 运行仿真，让模型控制机器人
for _ in range(10):  # 运行 100 个时间步
    action, _ = model.predict(obs)  # 让模型预测动作
    obs, reward, done, truncated, _ = env.step(action)  # 执行动作
    time.sleep(1)
    print(f"obs:{obs} Action: {action}, Reward: {reward}")  # 打印动作和奖励
    obs, _ = env.reset()  # 重新开始

    time.sleep(1)  # 控制仿真速度
# 关闭环境
env.close()
