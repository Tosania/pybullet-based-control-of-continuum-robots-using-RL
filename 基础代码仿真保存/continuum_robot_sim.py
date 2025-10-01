import pybullet as p
import pybullet_data
import time
import tkinter as tk
import threading 
from dataclasses import dataclass
from typing import List  # 需要导入 List

@dataclass
class arm_joint:
    in_control_joint: List[int]  # 存储数组
    first_control_joint: int         # 第一个数字
    direction: int      # 第二个数字（可以是浮点数）

class ContinuumRobotSim:
    def __init__(self, urdf_path="continuum_robot.urdf"):
        self.urdf_path = urdf_path
        self.robot_id = None
        self.num_joints = 0
        self.controlled_joints = []
        self.controlled_joints_arm=[arm_joint([],0,0),arm_joint([],0,0),arm_joint([],0,0)]
        self.in_base=0
        
        self.physics_client = p.connect(p.GUI)

        p.setGravity(0, 0, 0)

        self.load_robot()

    def load_robot(self):
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)

        print(f" 机器人加载完成，共有 {self.num_joints} 个关节")
        self.print_joint_info()
        self.print_arm_info()
        for i in range(p.getNumJoints(self.robot_id)):
          p.changeDynamics(self.robot_id, i, linearDamping=4, angularDamping=4)


    def print_arm_info(self):
       for i in range(0,3):
          print(f"the direction joint in arm{i+1} is:{self.controlled_joints_arm[i].direction}")
          print(f"the inside joint in arm{i+1} is:{self.controlled_joints_arm[i].first_control_joint} ",end='')
          for j in self.controlled_joints_arm[i].in_control_joint:
             print(j,end=" ")
          print()
          
    def print_joint_info(self):
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
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
        p.resetJointState(self.robot_id, jointIndex=joint_index,targetValue=target_positions) 
    
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
            for i in range(1,ling):
                self.set_joint_positions(self.controlled_joints_arm[index].in_control_joint[i],0)
            self.set_joint_positions(self.controlled_joints_arm[index].in_control_joint[ling],alpha/20.0)
            for i in range(ling+1,10):
              self.set_joint_positions(self.controlled_joints_arm[index].in_control_joint[i],alpha/10.0)
            
    def set_robot(self,theta_1,alpha_1,alpha_2,theta_2,alpha_3,theta_3,base_theta,length):
        self.in_base=(int)(-length*10)
        p.resetJointState(self.robot_id, jointIndex=1,targetValue=length)
        sim.set_arm(0,theta_1,alpha_1)
        sim.set_arm(1,alpha_2,theta_2)
        sim.set_arm(2,alpha_3,theta_3)
        sim.set_joint_positions(0,base_theta)
        
    def start_gui(self):
        root = tk.Tk()
        root.title("Continuum Robot Controller")
        self.theta_1 = tk.DoubleVar(value=0)
        self.alpha_1 = tk.DoubleVar(value=0)
        self.alpha_2 = tk.DoubleVar(value=0)
        self.theta_2 = tk.DoubleVar(value=0)
        self.alpha_3 = tk.DoubleVar(value=0)
        self.theta_3 = tk.DoubleVar(value=0)
        self.base_theta = tk.DoubleVar(value=0)
        self.length = tk.DoubleVar(value=0)
        
        def update_robot(*args):
            self.set_robot(
                self.theta_1.get(),
                self.alpha_1.get(),
                self.theta_2.get(),
                self.alpha_2.get(),
                self.theta_3.get(),
                self.alpha_3.get(),
                self.base_theta.get(),
                self.length.get()
            )
            
        sliders = [
            ("Theta 1", self.theta_1, -3.14, 3.14),
            ("Alpha 1", self.alpha_1, -3.14, 3.14),
            ("Theta 2", self.theta_2, -3.14, 3.14),
            ("Alpha 2", self.alpha_2, -3.14, 3.14),
            ("Theta 3", self.theta_3, -3.14, 3.14),
            ("Alpha 3", self.alpha_3, -3.14, 3.14),
            ("base_theta", self.base_theta, -3.14, 3.14),
            ("length", self.length, -2, 0),
        ]

        for text, var, min_val, max_val in sliders:
            tk.Label(root, text=text).pack()
            slider = tk.Scale(root, from_=min_val, to=max_val, resolution=0.01, orient="horizontal", variable=var, command=update_robot)
            slider.pack()
            
        root.mainloop()
        
    def run_simulation(self):
        while True:
            p.stepSimulation()
        p.disconnect()

if __name__ == "__main__":
    sim = ContinuumRobotSim()
    sim_thread = threading.Thread(target=sim.run_simulation, daemon=True)
    sim_thread.start()
    sim.start_gui()
