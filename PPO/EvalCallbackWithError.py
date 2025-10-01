from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv
import numpy as np
import os

class EvalCallbackWithError(BaseCallback):
    def __init__(self, target_path="reachable_targets_easy.npy", eval_freq=100000, log_dir="./ppo_log/", name="avg_error", verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.name = name
        self.target_points = np.load(target_path)
        self.writer = SummaryWriter(log_dir)
        self.batch_size = 24

    def _init_callback(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_error = self.evaluate_model()
            if self.verbose > 0:
                print(f"\n📌 [Step {self.n_calls}] 平均误差: {mean_error:.4f}")
            self.writer.add_scalar(self.name, mean_error, self.n_calls)
        return True

    def evaluate_model(self):
        all_errors = []
        for i in range(0, len(self.target_points), self.batch_size):
            batch_targets = self.target_points[i:i+self.batch_size]

            def make_env(index, target):
                def _init():
                    env = ContinuumRobotEnv(render_mode=False, control_mode=1)
                    env.sim.keshi_ball(target)
                    return env
                return _init

            env_fns = [make_env(idx, pt) for idx, pt in enumerate(batch_targets)]
            vec_env = SubprocVecEnv(env_fns)
            obs = vec_env.reset()
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)
            end_positions = vec_env.env_method("sim.get_end_effector_position")

            for j, end_pos in enumerate(end_positions):
                error = np.linalg.norm(np.array(end_pos) - np.array(batch_targets[j]))
                all_errors.append(error)

            vec_env.close()
        return np.mean(all_errors)
