from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv
import torch

policy_kwargs = dict(
    net_arch=[1024, 1024, 512],  # 🧠 超大网络结构
)
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # 可选，但建议加上
# 并行环境：根据 CPU 核心来选（推荐 8～16）
    env = make_vec_env(
        ContinuumRobotEnv,
        n_envs=24,  # 并行 8 个环境（可调）
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
        "control_mode": 1,      # ✅ 你要传的变量
        "render_mode": False
        }
    )

    # 使用 GPU + 大 batch
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        policy_kwargs=policy_kwargs,
        n_steps=4096,         # 每个环境采样 4096 步 → 8×4096 = 32768 样本
        batch_size=16384,     # 每次训练 16384 样本（吃掉大显存）
        n_epochs=10,          # 多轮训练
        verbose=0,
        tensorboard_log="./ppo_log/"
    )

    model.learn(total_timesteps=100000, progress_bar=True)
    model.save("continuum_robot_ppo")
    env.close()
