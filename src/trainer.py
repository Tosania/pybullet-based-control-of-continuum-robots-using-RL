from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
from stable_baselines3.common.vec_env import VecNormalize
from call_back import SaveBestValueLossCallback
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
        title,
        model_type="PPO",
        total_timesteps=400_000,
        n_envs=24,
        device="cuda",
        reward_fn=None,
        buffer_size=1_000_000,
        done_fn=None,
        policy_kwargs=None,
        control_mode=1,
        render_mode=False,
        log_path="./ppo_log/",
        model_path="./model/continuum_robot_ppo",
        learning_rate=3e-4,
        batch_size=8192*2,
        n_steps=2048*2,
        train_freq=1,
        learning_starts=10000,
        n_epochs=10,
        seed=0,
        verbose=0,
        generate_pdf=1,
        check_freq=10,
    ):
        self.callback = SaveBestValueLossCallback(
            verbose=0,
            check_freq=check_freq,
            save_path=f"{model_path}_best",
            target_path="./source/reachable_targets_easy.npy",
            device=device,
            control_mode=control_mode
        )
        self.title=title
        self.train_freq=train_freq
        self.learning_starts=learning_starts
        self.model_type=model_type
        self.reward_fn=reward_fn
        self.control_mode=control_mode
        self.done_fn=done_fn
        self.total_timesteps = total_timesteps
        self.seed=seed
        self.device = device
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.n_steps=n_steps
        self.n_epochs=n_epochs
        self.n_env=n_envs
        self.buffer_size=buffer_size
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

        new_logger = configure(log_path, ["tensorboard"])
        if self.model_type=="PPO":
          self.env = make_vec_env(
              ContinuumRobotEnv,
              n_envs=n_envs,
              vec_env_cls=SubprocVecEnv,
              env_kwargs=env_kwargs
          )
          self.model = PPO(
              policy="MlpPolicy",
              env=self.env,
              device=device,
              policy_kwargs=policy_kwargs,
              learning_rate=learning_rate,
              seed=self.seed,
              n_steps=n_steps,
              batch_size=batch_size,
              n_epochs=n_epochs,
              verbose=verbose
          )
        elif self.model_type=="SAC":
            self.env = make_vec_env(
            ContinuumRobotEnv,
            n_envs=n_envs,  # 减少并行环境数量
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs
          )
            self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
            self.model = SAC(
              policy="MlpPolicy",
              env=self.env,
              device=device,
              seed=seed,
              policy_kwargs=policy_kwargs,
              buffer_size=buffer_size,
              train_freq=train_freq,
              learning_starts=learning_starts,
              learning_rate=learning_rate,
              batch_size=batch_size,  # 减小批次大小
              verbose=verbose
          )
        self.model.set_logger(new_logger)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=True,callback=self.callback)
        self.model.save(self.model_path)
        self.judge=ModelEvaluator(model_path=f"{self.model_path}_best", target_path="./source/reachable_targets_easy.npy", device="cuda", control_mode=1, render=False,model_type=self.model_type)
        self.error=self.judge.evaluate()
        self.env.close()
        if self.g_pdf==1:
            self.pdf=ReportGenerator(
                log_dir=self.log_path,
                model_type=self.model_type,
                model_name=self.model_path,
                seed=self.seed,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                n_steps=self.n_steps,
                n_env=self.n_env,
                n_epochs=self.n_epochs,
                total_timesteps=self.total_timesteps,
                control_mode = self.control_mode,
                reward_fn=self.reward_fn,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_starts,
                train_freq=self.train_freq,
                done_fn=self.done_fn,
                device=self.device,
                net_arch=self.model.policy_kwargs.get("net_arch"),
                output_path=f"./pdf/{self.title}_report.pdf",
                judge=self.error,
                best_reward=self.callback.best_reward
                )
            self.pdf.generate_pdf()