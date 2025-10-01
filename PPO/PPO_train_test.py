from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import multiprocessing
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv
import torch
from torch import nn
torch.backends.cudnn.benchmark = True
policy_kwargs = dict(
    net_arch=[1024, 1024, 512],  # 🧠 超大网络结构
)
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # 可选，但建议加上
# 并行环境：根据 CPU 核心来选（推荐 8～16）
    env = make_vec_env(
        ContinuumRobotEnv,
        n_envs=30,  # 并行 8 个环境（可调）
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
        "control_mode": 0,      # ✅ 你要传的变量
        "render_mode": False
        }
    )

    # 使用 GPU + 大 batch
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cuda",
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        n_steps=512,         # 每个环境采样 4096 步 → 8×4096 = 32768 样本
        batch_size=1024*3,     # 每次训练 16384 样本（吃掉大显存）
        n_epochs=4,          # 多轮训练        verbose=1,
                # ✅ 限制 value 函数更新
        tensorboard_log="./ppo_log/",
        verbose=0
    )
    #model = PPO.load("continuum_robot_ppo_2e7", env=env, device="cuda")
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("continuum_robot_ppo_step")
    env.close()
