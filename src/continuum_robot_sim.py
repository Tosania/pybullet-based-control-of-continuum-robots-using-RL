import pybullet as p
import pybullet_data
import time
import tkinter as tk
import threading 
from dataclasses import dataclass
from typing import List  # 需要导入 List
import random

@dataclass
class arm_joint:
    in_control_joint: List[int]  # 存储数组
    first_control_joint: int         # 第一个数字
    direction: int      # 第二个数字（可以是浮点数）

class ContinuumRobotSim:
    def __init__(self, urdf_path="continuum_robot.urdf",connection_mode=p.DIRECT,controlmode=1):
        self.urdf_path = urdf_path
        self.control_mode=controlmode          #1表示位置控制，0表示速度控制
        self.robot_id = None
        self.num_joints = 0
        self.running=True
        self.controlled_joints = []
        self.controlled_joints_arm=[arm_joint([],0,0),arm_joint([],0,0),arm_joint([],0,0)]
        self.in_base=0
        self.theta_1=0
        self.alpha_1=0
        self.alpha_2=0
        self.theta_2=0
        self.alpha_3=0
        self.theta_3=0
        self.base_theta=0
        self.length=0
        self.delta_action=[0,0,0,0,0,0,0,0]
        self.physics_client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)

        self.load_robot()
    def keshi_ball(self,position):
    # 清除旧的可视化目标点（如果有）
        if hasattr(self, "sphere_id"):  # 检查是否已有小球
          p.removeBody(self.sphere_id)  # ✅ 删除旧小球
    
        radius = 0.1  # 小球半径
        mass = 0  # 质量
        start_pos = position  # 随机位置
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
        col_sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_sphere_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])  # 红色球
    
        self.sphere_id = p.createMultiBody(mass, col_sphere_id, vis_sphere_id, start_pos, start_orientation)  # ✅ 存储新小球 ID

    def load_robot(self):
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.physics_client)
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)

        print(f" 机器人加载完成，共有 {self.num_joints} 个关节")
        self.print_joint_info()
        self.print_arm_info()
        for i in range(p.getNumJoints(self.robot_id)):
          p.changeDynamics(self.robot_id, i, linearDamping=4, angularDamping=4, physicsClientId=self.physics_client)
          
    def get_end_effector_position(self):
        end_effector_index = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client) - 1
        link_state = p.getLinkState(self.robot_id, end_effector_index, physicsClientId=self.physics_client)
        position = link_state[0]
        return position

    def print_arm_info(self):
       for i in range(0,3):
          print(f"the direction joint in arm{i+1} is:{self.controlled_joints_arm[i].direction}")
          print(f"the inside joint in arm{i+1} is:{self.controlled_joints_arm[i].first_control_joint} ",end='')
          for j in self.controlled_joints_arm[i].in_control_joint:
             print(j,end=" ")
          print()
          
    def print_joint_info(self):
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8') if isinstance(joint_info[1], bytes) else joint_info[1]
            print(f"Joint {i}: {joint_name} - Type: {joint_info[2]}")
            
            if joint_name.startswith("joint_0_0"):
              self.controlled_joints_arm[0].direction=i
            elif joint_name.startswith=="joint_0_1":
              self.controlled_joints_arm[1].direction=i
            elif joint_name=="joint_1_1":
              self.controlled_joints_arm[0].first_control_joint=i
            elif joint_name.startswith("joint_0_"):
              self.controlled_joints_arm[0].in_control_joint.append(i)
              
            if joint_name.startswith("joint_1_0"):
              self.controlled_joints_arm[1].direction=i
            elif joint_name=="joint_1_1":
              self.controlled_joints_arm[1].first_control_joint=i
            elif joint_name.startswith("joint_1_"):
              self.controlled_joints_arm[1].in_control_joint.append(i)
              
            if joint_name.startswith("joint_2_0"):
              self.controlled_joints_arm[2].direction=i
            elif joint_name=="joint_2_1":
              self.controlled_joints_arm[2].first_control_joint=i
            elif joint_name.startswith("joint_2_"):
              self.controlled_joints_arm[2].in_control_joint.append(i)

    def set_joint_positions(self,joint_index, target_positions):
        p.resetJointState(self.robot_id, jointIndex=joint_index,targetValue=target_positions, physicsClientId=self.physics_client)

    def set_arm(self,index,theta,alpha):
        self.set_joint_positions(self.controlled_joints_arm[index].direction,theta)
        if index*10+10<=self.in_base:
            for i in self.controlled_joints_arm[index].in_control_joint:
              self.set_joint_positions(i,0)
            self.set_joint_positions(self.controlled_joints_arm[index].first_control_joint,0)
        elif index*10>=self.in_base:
            self.set_joint_positions(self.controlled_joints_arm[index].first_control_joint,alpha/20.0)
            for i in self.controlled_joints_arm[index].in_control_joint:
              self.set_joint_positions(i,alpha/10.0)
        else:
            ling=self.in_base%10
            self.set_joint_positions(self.controlled_joints_arm[index].first_control_joint,0)
            for i in range(0,ling-1):
                self.set_joint_positions(self.controlled_joints_arm[index].in_control_joint[i],0)
            self.set_joint_positions(self.controlled_joints_arm[index].in_control_joint[ling-1],alpha/20.0)
            for i in range(ling,9):
              self.set_joint_positions(self.controlled_joints_arm[index].in_control_joint[i],alpha/10.0)
    def update_velocity(self,delta_action):
       self.delta_action[0]=delta_action[0]
       self.delta_action[1]=delta_action[1]
       self.delta_action[2]=delta_action[2]
       self.delta_action[3]=delta_action[3]
       self.delta_action[4]=delta_action[4]
       self.delta_action[5]=delta_action[5]
       self.delta_action[6]=delta_action[6]
       self.delta_action[7]=delta_action[7]
       self.apply_velocity_control()
       #p.stepSimulation(physicsClientId=self.physics_client)
    def apply_velocity_control(self, time_step=0.05):
        """
        用于速度控制模式。
        delta_action: 8个关节的速度
        time_step: 每一步的仿真时间间隔（默认 0.05s）
        """
        delta_action=self.delta_action
        self.theta_1 += delta_action[0] * time_step
        while self.theta_1>3.14:
           self.theta_1-=6.28
        while self.theta_1<-3.14:
          self.theta_1+=6.28
        self.alpha_1 += delta_action[1] * time_step
        self.alpha_1=min(3.14,max(self.alpha_1,-3.14))
        self.theta_2 += delta_action[2] * time_step
        while self.theta_2>3.14:
           self.theta_2-=6.28
        while self.theta_2<-3.14:
          self.theta_2+=6.28
        self.alpha_2 += delta_action[3] * time_step
        self.alpha_2=min(3.14,max(self.alpha_2,-3.14))
        self.theta_3 += delta_action[4] * time_step
        while self.theta_3>3.14:
           self.theta_3-=6.28
        while self.theta_3<-3.14:
          self.theta_3+=6.28
        self.alpha_3 += delta_action[5] * time_step
        self.alpha_3=min(3.14,max(self.alpha_3,-3.14))
        self.base_theta += delta_action[6] * time_step
        while self.base_theta>3.14:
           self.base_theta-=6.28
        while self.base_theta<-3.14:
          self.base_theta+=6.28
        self.length += delta_action[7] * time_step
        self.length=min(0,max(self.length,-2))
        self.set_robot(self.theta_1,self.alpha_1,self.theta_2,self.alpha_2,self.theta_3,self.alpha_3,self.base_theta,self.length)
    def set_robot(self,theta_1,alpha_1,theta_2,alpha_2,theta_3,alpha_3,base_theta,length):
        #print("out_line:",theta_3,alpha_3)
        self.theta_1=theta_1
        self.alpha_1=alpha_1
        self.alpha_2=alpha_2
        self.theta_2=theta_2
        self.alpha_3=alpha_3
        self.theta_3=theta_3
        self.base_theta=base_theta
        self.length=length
        self.in_base=(int)(-length*10)
        #p.setJointMotorControl2(self.robot_id, jointIndex=1, controlMode=p.POSITION_CONTROL, targetPosition=length)
        self.set_arm(0,theta_1,alpha_1)
        self.set_arm(1,theta_2,alpha_2)
        self.set_arm(2,theta_3,alpha_3)
        self.set_joint_positions(0,base_theta)
        p.resetJointState(self.robot_id, jointIndex=1,targetValue=length, physicsClientId=self.physics_client)
        p.stepSimulation(physicsClientId=self.physics_client)
    def getstate(self):
       return [self.theta_1,self.alpha_1,self.theta_2,self.alpha_2,self.theta_3,self.alpha_3,self.base_theta,self.length]
    def run_simulation(self):
        while self.running:
            #if self.control_mode==0:
            #self.apply_velocity_control()
            p.stepSimulation(physicsClientId=self.physics_client)
        #p.disconnect()
    
    def close(self):
      self.running = False  # ✅ 停止仿真
      p.disconnect()
if __name__ == "__main__":
    sim=ContinuumRobotSim()