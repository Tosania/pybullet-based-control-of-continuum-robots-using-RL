import numpy as np
import os
import sys
from stable_baselines3 import PPO

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv

class ModelEvaluator:
    def __init__(self, model_path, target_path, device="cuda", control_mode=1, render=False):
        self.model = PPO.load(model_path, device=device)
        self.targets = np.load(target_path)
        self.env = ContinuumRobotEnv(render_mode=render, control_mode=control_mode)
        self.errors = []

    def evaluate(self):
        self.errors.clear()
        for idx, target in enumerate(self.targets):
            self.env.sim.keshi_ball(target)
            self.env.ball = target

            obs = self.env._get_observation()
            action, _ = self.model.predict(obs, deterministic=True)
            self.env.sim.set_robot(*action)

            pred_pos = self.env.sim.get_end_effector_position()
            error = np.linalg.norm(np.array(pred_pos) - np.array(target))
            self.errors.append(error)

        mean_error = np.mean(self.errors)
        return mean_error
judge=ModelEvaluator(model_path="test_0_best", target_path="reachable_targets_easy.npy", device="cuda", control_mode=1, render=False)
print(judge.evaluate())