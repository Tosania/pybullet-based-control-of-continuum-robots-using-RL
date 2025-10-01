import os

# 生成 URDF 文件的路径
urdf_filename = "continuum_robot.urdf"

# 设定参数
num_links = 10  # 生成的 link 数量
link_length = 0.1  # 每个 link 的长度
link_radius = 0.05
joint_axis = "1 0 0"  # 关节旋转轴（沿 X 轴）
base_length= 2
wai_length=0.002
wai_radius=0.2

# 生成 URDF 代码
urdf_content = """<?xml version="1.0"?>
<robot name="continuum_robot">
"""

# 添加底座 link
urdf_content += f"""
  <link name="base_link">
    <visual>
      <origin xyz="0 0 {base_length/2}" rpy="0 0 0"/>
      <geometry>
        <box size="0.6 0.6 {base_length}"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    
  </link>
  <link name="extend_link">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.00001"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
  </link>

  <!-- 软体部分的滑动关节 -->
  <joint name="extend_joint" type="prismatic">
    <parent link="base_link"/>
    <child link="extend_link"/>
    <origin xyz="0 0 {base_length}"/>
    <axis xyz="0 0 1"/>  <!-- 让软体部分沿 Z 轴伸缩 -->
    <limit lower="{-base_length}" upper="0" effort="50.0" velocity="2.0"/>
  </joint>
  
  <link name="base_rotation">
    <visual>
      <origin xyz="0 0 0.00000005"/>
      <geometry>
        <cylinder radius="0.000001" length="0.0000001"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
  </link>
"""
def generate(last_name,num):
  global urdf_content
  urdf_content += f"""<link name="link_{num}_help">
      <visual>
        <origin xyz="0 0 0.00015" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.0003"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.0003"/>
        </geometry>
      </collision>
    </link>
  
    <joint name="joint_{num}_0" type="continuous">
      <parent link="{last_name}"/>
      <child link="link_{num}_help"/>
      <origin xyz="0 0 {link_length if num!=0 else 0.0005}"/>
      <axis xyz="0 0 1"/>
      <limit effort="10.0" velocity="2.0"/>
    </joint>
  """

  # 生成多个 link 和 continuous 关节
  for i in range(1, num_links + 1):
      urdf_content += f"""
    <link name="link_{num}_{i}">
      <visual>
        <origin xyz="0 0 {link_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="{link_radius}" length="{link_length}"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="{link_radius}" length="{link_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
      </inertial> <link name="link_{num}_{i}">
      <visual>
        <origin xyz="0 0 {link_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="{link_radius}" length="{link_length}"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="{link_radius}" length="{link_length}"/>
        </geometry>
      </collision>
    </link>
    </link>

    <joint name="joint_{num}_{i}" type="continuous">
      <parent link="{f'link_{num}_help' if i == 1 else f'link_{num}_{i-1}'}"/>
      <child link="link_{num}_{i}"/>
      <origin xyz="0 0 {link_length if i !=1 else 0.00015}"/>
      <axis xyz="{joint_axis}"/>
      <limit effort="10.0" velocity="2.0"/>
    </joint>"""
      urdf_content += f""" <link name="linkwai_{num}_{i}">
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="{wai_radius}" length="{wai_length}"/>
        </geometry>
        <material name="blask">
          <color rgba="0 0 0 0.5"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="{wai_radius}" length="{wai_length}"/>
        </geometry>
      </collision>
    </link>
     <joint name="jointwai_{num}_{i}" type="fixed">
      <parent link="link_{num}_{i}"/>
      <child link="linkwai_{num}_{i}"/>
      <origin xyz="0 0 {link_length/2}"/>
    </joint>
  """
generate("extend_link",0)
generate("link_0_10",1)
generate("link_1_10",2)
# 结束 URDF 结构
urdf_content+=f"""<!-- 添加底座旋转关节 -->
  <joint name="base_joint" type="continuous">
    <parent link="base_rotation"/>
    <child link="base_link"/>
    <origin xyz="0 0 0.0000001"/>
    <axis xyz="0 1 0"/>  <!-- 让底座绕 y 轴旋转 -->
    <limit effort="50.0" velocity="5.0"/>
  </joint>"""
urdf_content += "\n</robot>"

# 写入文件
with open(urdf_filename, "w", encoding="utf-8") as f:
    f.write(urdf_content)

print(f"✅ URDF 文件已生成: {urdf_filename}")
1