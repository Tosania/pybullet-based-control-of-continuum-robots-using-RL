from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from trainer import ContinuumTrainer
import numpy as np
import torch
import os
import gc

def my_custom_reward(distance):
    return -distance + np.exp(-distance**2) * 5

def my_custom_done(reward, step,distance,in_step):
    return reward>=-0.5

def my_custom_done1(reward, step,distance,in_step):
    return distance<=1

def train_one_experiment(
    title="test_1",
    model_type="PPO",
    total_timesteps=1000,
    model_path="./model/test_1",
    log_path="./ppo_log/test_1",
    control_mode=1,
    n_envs=24,
    device="cuda",
    learning_rate=3e-4,
    batch_size=4096*4,
    buffer_size=1000000,
    seed=0,
    learning_starts=10000,
    train_freq=1,
    n_steps=2048,
    n_epochs=10,
    verbose=0,
    reward_fn=my_custom_reward,
    done_fn=my_custom_done,
    generate_pdf=0,
    check_freq=1000
):
    # ✅ 自定义网络结构
    policy_kwargs = dict(net_arch=[1024, 1024, 512])

    trainer = ContinuumTrainer(
        title=title,
        model_type=model_type,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        device=device,
        policy_kwargs=policy_kwargs,
        control_mode=control_mode,
        render_mode=False,
        log_path=log_path,
        model_path=model_path,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        seed=seed,
        n_steps=n_steps,
        n_epochs=n_epochs,
        verbose=verbose,
        reward_fn=reward_fn,
        done_fn=done_fn,
        generate_pdf=generate_pdf,
        check_freq=check_freq
    )

    trainer.train()

    # ✅ 显式释放资源
    del trainer
    gc.collect()

if __name__ == "__main__":
    from multiprocessing import Process

    p2 = Process(target=train_one_experiment, kwargs={
        "title":"test_judge_1",
        "model_type":"PPO",
        "total_timesteps":6000000,
        "model_path":"./model/test_judge_1",
        "log_path":"./ppo_log/test_judge_1",
        "control_mode":1,
        "n_envs":11,
        "device":"cuda",
        "seed":1,
        "learning_rate":3e-4,
        "batch_size":20000,
        "buffer_size":100000,
        "learning_starts":20000,
        "train_freq":4,
        "n_steps":2048*2,
        "n_epochs":10,
        "verbose":0,
        "reward_fn":my_custom_reward,
        "done_fn":my_custom_done,
        "generate_pdf":1,
        "check_freq":1000
    })
    p2.start()
    p2.join()
