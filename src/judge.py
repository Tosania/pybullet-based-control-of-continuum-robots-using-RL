import numpy as np
import os
import sys
from stable_baselines3 import PPO, SAC
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gym_continuum_env import ContinuumRobotEnv

class ModelEvaluator:
    def __init__(self, model_path, target_path, device="cuda", control_mode=1, render=False, model_type="PPO"):
        self.device = device
        self.control_mode = control_mode
        self.render = render
        self.model_type = model_type
        
        self.env = ContinuumRobotEnv(render_mode=render, control_mode=control_mode)
        print("✅ env observation_space:", self.env.observation_space)
        print("✅ env action_space:", self.env.action_space)
        # 根据模型类型加载不同的模型
        if model_type == "PPO":
            self.model = PPO.load(
                model_path, 
                custom_objects={
        "observation_space": self.env.observation_space,
        "action_space": self.env.action_space
    },
                device=device,
            )
        elif model_type == "SAC":
            self.model = SAC.load(
                model_path,
                device=device,
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        self.targets = np.load(target_path)
        self.errors = []

    def evaluate(self):
        """评估模型在所有目标点上的平均误差"""
        self.errors.clear()
        self.num=0
        for target in self.targets:
            self.env.ball = target
            if target[0]*target[0]+target[1]*target[1]+target[2]*target[2]<=3*3:
              for i in range(10000):
                obs = self.env._get_observation()
                action, _ = self.model.predict(obs, deterministic=True)
                self.num+=1
                self.env.sim.update_velocity(action)
                pred_pos = self.env.sim.get_end_effector_position()
                error = np.linalg.norm(np.array(pred_pos) - np.array(target))
                #print(error,end=" ")
              self.errors.append(error)
              break
        
        mean_error = np.mean(self.errors)
        print(self.num)
        print(f"平均误差: {mean_error:.4f}")
        return mean_error
if __name__ == "__main__": 
  judge=ModelEvaluator(model_path="new_reward_2_best_step", target_path="./source/reachable_targets.npy", device="cuda", control_mode=0, render=False,model_type="PPO")
  distance=judge.evaluate()
  