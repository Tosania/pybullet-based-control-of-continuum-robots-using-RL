import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import threading
import time
import math
import sys
import random
sys.path.append("./sim")
from continuum_robot_sim import ContinuumRobotSim

class ContinuumRobotEnv(gym.Env):
    """ 自定义 OpenAI Gym 环境，控制连续体机器人 """
    
    def __init__(self, render_mode=False, env_id=0, control_mode="full"):
        super(ContinuumRobotEnv, self).__init__()
        self.env_id = env_id
        self.control_mode = control_mode
        #self.sim = ContinuumRobotSim(connection_mode=p.DIRECT)
        # 机器人仿真
        self.sim = ContinuumRobotSim()
        self.ball_id=None
        # 关节角度范围（动作空间）
        if self.control_mode == "full":
            self.action_space = spaces.Box(
                low=np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -2]),
                high=np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0]),
                #low=np.array([0,0,0,0,-3.14,1.57/2.0,0,0]),
                #high=np.array([0,0,0,0,3.14,1.57/2.0,0,0]),
                dtype=np.float32
            )
        elif self.control_mode == "step":
            self.action_space = spaces.Box(
                low=np.array([-0.3, -0.3,-0.3, -0.3,-0.3, -0.3, -0.3,-0.3]),
                high=np.array([0.3, 0.3, 0.3, 0.3,0.3, 0.3, 0.3,0.3]),
                dtype=np.float32
            )
        # 观察空间（机器人末端位置 + 小球目标位置）
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -5, -5, -2.5, -5]),
            high=np.array([5, 5, 5, 5, 2.5, 5]),
            dtype=np.float32
        )

        # 启动仿真线程
        #self.sim_thread = threading.Thread(target=self.sim.run_simulation, daemon=True)
        #self.sim_thread.start()


        # 休眠等待仿真稳定
        #time.sleep(1)
    def sample_point_in_sphere(self,radius=2.5, center=[0, 0, 0]):
      vec = np.random.normal(0, 1, 3)
      vec /= np.linalg.norm(vec)

      r = radius * np.cbrt(np.random.uniform(0, 1))

      return center[0] + r * vec[0], center[1] + r * vec[1], center[2] + r * vec[2]
    def reset(self, seed=None, options=None):
        """ 重置环境，返回初始观察值 """
        super().reset(seed=seed)
        #self.sim.set_robot(
        #0,0,0,0,0,0,0,0
    #)

        self.ball = self.sample_point_in_sphere()
        self.num_step=0
        self.sim.keshi_ball(self.ball)
        return self._get_observation(), {}

    def step(self, action):
        """ 执行动作，返回 (obs, reward, done, info) """

        # 设置机器人关节
        if self.control_mode == "full":
            self.sim.set_robot(*action)
        else:
            raise ValueError("Unsupported control_mode!")
        #time.sleep(0.0005)
        # 计算奖励
        reward = self._calculate_reward()
        self.num_step+=1
        # 检查是否完成
        done = self.num_step>=1000  # 当机器人足够接近小球时，任务完成
        truncated = False  # ✅ Gym 0.26+ 需要 `truncated`，这里默认为 False

        return self._get_observation(), reward, done, truncated, {}

    def _get_observation(self):
        """ 获取机器人末端和目标小球的位置 """
        end_effector_pos = self.sim.get_end_effector_position()
        ball_pos=self.ball

        # 观察空间：机器人末端位置 + 小球位置
        return np.array([
            end_effector_pos[0], end_effector_pos[1], end_effector_pos[2],
            ball_pos[0], ball_pos[1], ball_pos[2]
        ], dtype=np.float64)

    def _calculate_reward(self):
        """ 计算奖励函数，目标是最小化末端到小球的距离 """
        end_effector_pos = self.sim.get_end_effector_position()
        ball_pos=self.ball
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(ball_pos))
        reward = -np.square(distance)*1.0
        if np.isclose(distance, 0.0, atol=0.05 * 2.0).all():
          reward += 0.5

        if np.isclose(distance, 0.0, atol=0.05).all():
          reward += 1.5
        
        # 奖励 = -距离（越接近小球，奖励越高）
        return reward

    def render(self, mode="human"):
        """ 可视化（如果需要额外的渲染逻辑，可以在这里实现） """
        pass

    def close(self):
        """ 关闭环境 """
        self.sim.running = False
