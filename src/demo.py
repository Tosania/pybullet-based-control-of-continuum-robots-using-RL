from stable_baselines3 import PPO
import pybullet as p
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv
import keyboard
import numpy as np
import time

# 加载模型
env = ContinuumRobotEnv(render_mode=False, control_mode=1,connection_mode=p.GUI)
model = PPO.load(
        "new_reward_2_best", 
        custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space
    },
                device="cuda",
            )

# 初始化环境
obs, _ = env.reset()

print("按下【空格】→ 执行一次预测\n按下【K】→ 重置环境\n按下【ESC】→ 退出程序")

while True:
    if keyboard.is_pressed("space"):
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(f"预测动作: {np.round(action, 3)}, 奖励: {round(reward, 3)}")
        while keyboard.is_pressed("space"):  # 防止重复触发
            time.sleep(0.1)
    elif keyboard.is_pressed("k"):
        obs, _ = env.reset()
        print("环境已重置")
        while keyboard.is_pressed("k"):
            time.sleep(0.1)
    elif keyboard.is_pressed("esc"):
        print("退出程序")
        break

env.close()
