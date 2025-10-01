from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_continuum_env import ContinuumRobotEnv
from pdf_generate import ReportGenerator
import torch
import os
import sys
import multiprocessing
from judge import ModelEvaluator
class ContinuumTrainer:
    def __init__(
        self,
        total_timesteps=400_000,
        n_envs=24,
        device="cuda",
        reward_fn=None,
        done_fn=None,
        policy_kwargs=None,
        control_mode=1,
        render_mode=False,
        log_path="./ppo_log/",
        model_path="continuum_robot_ppo_step",
        learning_rate=3e-4,
        batch_size=8192*2,
        n_steps=2048*2,
        n_epochs=10,
        verbose=0,
        generate_pdf=1
    ):
        self.reward_fn=reward_fn
        self.done_fn=done_fn
        self.total_timesteps = total_timesteps
        self.device = device
        self.model_path = model_path
        self.log_path = log_path
        self.g_pdf=generate_pdf
        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[1024, 1024, 512])

        env_kwargs = {
            "control_mode": control_mode,
            "render_mode": render_mode,
            "reward_fn": reward_fn,
            "done_fn": done_fn
        }

        multiprocessing.set_start_method("spawn", force=True)

        self.env = make_vec_env(
            ContinuumRobotEnv,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs
        )

        new_logger = configure(log_path, ["tensorboard"])
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            device=device,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=verbose
        )
        self.model.set_logger(new_logger)
    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=True)
        self.model.save(self.model_path)
        self.judge=ModelEvaluator(model_path=self.model_path, target_path="reachable_targets_easy.npy", device="cuda", control_mode=1, render=False)
        self.error=self.judge.evaluate()
        self.env.close()
        if self.g_pdf==1:
            self.pdf=ReportGenerator(
                log_dir=self.log_path,
                model_name=self.model_path,
                total_timesteps=self.total_timesteps,
                control_mode = 1,
                reward_fn=self.reward_fn,
                done_fn=self.done_fn,
                device=self.device,
                net_arch=self.model.policy_kwargs.get("net_arch"),
                output_path=f"{self.model_path}_report.pdf",
                judge=self.error
                )
            self.pdf.generate_pdf()