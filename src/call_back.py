from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from gym_continuum_env import ContinuumRobotEnv
import threading
import queue

class SaveBestValueLossCallback(BaseCallback):
    def __init__(self, check_freq, save_path, target_path, device="cuda", control_mode=1, verbose=1):
        super(SaveBestValueLossCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_error = float("inf")
        self.target_path = target_path
        self.device = device
        self.control_mode = control_mode
        self.best_reward = float("-inf")

    def _on_step(self):
        # 每 check_freq 步获取一次奖励
        if self.n_calls % self.check_freq == 0:
            
            # 获取最新的平均奖励
            if hasattr(self.model.env, 'get_attr'):
                mean_rewards = self.model.env.get_attr('mean_rewards')
                if len(mean_rewards) > 0:
                    # 计算所有环境的平均奖励
                    overall_mean_reward = np.mean(mean_rewards)
                    self.logger.record("train/mean_reward", overall_mean_reward)
                    
                    # 更新最佳奖励并保存模型
                    if overall_mean_reward > self.best_reward:
                        self.best_reward = overall_mean_reward
                        if self.verbose > 0:
                            print(f"📈 新的最高奖励: {overall_mean_reward:.4f}，正在保存模型...")
                        self.model.save(self.save_path)
        
        return True