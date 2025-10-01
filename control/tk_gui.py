import tkinter as tk
import sys
import pybullet as p
import threading
sys.path.append("./sim")
from continuum_robot_sim import ContinuumRobotSim
class tk_gui():
  def __init__(self):
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
          self.end_effector_label = tk.Label(root, text="末端位置: (0, 0, 0)", font=("Arial", 12))
          self.ball_label = tk.Label(root, text="小球位置: (0, 0, 0)", font=("Arial", 12))
          self.end_effector_label.pack()
          self.ball_label.pack()
          self.sim = ContinuumRobotSim(connection_mode=p.GUI,controlmode=0)
          #sim_thread = threading.Thread(target=self.sim.run_simulation, daemon=True)
          #sim_thread.start()
          def update_robot(*args):
              self.sim.update_velocity(
                  [self.theta_1.get(),
                  self.alpha_1.get(),
                  self.theta_2.get(),
                  self.alpha_2.get(),
                  self.theta_3.get(),
                  self.alpha_3.get(),
                  self.base_theta.get(),
                  self.length.get()]
              )
              self.sim.apply_velocity_control()
              
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
          def update_end_effector_position():
            position = self.sim.getstate()
            #position2, _ = p.getBasePositionAndOrientation(self.sphere_id)
            position2=[0,0,0]
            self.end_effector_label.config(text=f"状态: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f},{position[3]:.3f}, {position[4]:.3f}, {position[5]:.3f}),{position[6]:.3f}, {position[7]:.3f}")
            self.ball_label.config(text=f"小球位置: ({position2[0]:.3f}, {position2[1]:.3f}, {position2[2]:.3f})")
            root.after(100, update_end_effector_position)  # 每100ms 更新一次
          update_end_effector_position()  
          
          root.mainloop()
if __name__ == "__main__":
    tk_gui()