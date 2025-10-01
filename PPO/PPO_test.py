from stable_baselines3 import PPO
import pybullet as p
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv
import numpy as np

# 加载训练好的模型
model=PPO.load("continuum_robot_ppo_2e7")
env = ContinuumRobotEnv(control_mode=1,connection_mode=p.DIRECT)
obs, _ = env.reset()
total_rewards1 = []
total_rewards2 = []
for episode in range(100):  # 运行10次实验
    obs, _ = env.reset()
    total_reward1 = 0
    total_reward2 = 0
    action = env.action_space.sample()  # 采样随机动作
    obs, reward, done, _, _ = env.step(action)
    total_reward1 += reward
    action, _ = model.predict(obs)  # 用训练好的模型预测动
    obs, reward, done, _, _ = env.step(action)
    total_reward2 += reward
    total_rewards1.append(total_reward1)
    total_rewards2.append(total_reward2)

print("随机策略平均得分:", np.mean(total_rewards1))
print("训练策略平均得分:", np.mean(total_rewards2))
env.close()
