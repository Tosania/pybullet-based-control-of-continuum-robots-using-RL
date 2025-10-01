from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
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
from trainer import ContinuumTrainer
def my_custom_reward(distance):
    reward = -distance + np.exp(-distance**2) * 5
    return reward
def my_custom_done1(reward,step):
    #done=reward > -0.5
    done=step>3
    return done
def my_custom_done2(reward,step):
    #done=reward > -0.5
    done=step>3
    return done
if __name__ == "__main__":
    trainer = ContinuumTrainer(
        total_timesteps=100,
        n_envs=2,
        device="cuda",
        policy_kwargs=policy_kwargs,
        control_mode=1,
        render_mode=False,
        log_path="./ppo_log/test1",
        model_path="test_1",
        learning_rate=3e-4,
        batch_size=2,
        n_steps=10,
        n_epochs=10,
        verbose=0,
        reward_fn=my_custom_reward,
        done_fn=my_custom_done1
    )
    trainer.train()