import os

def generate_cup_urdf(width_cm, height_cm, filename=None):
    if filename is None:
        filename = f"./objects_new/cup_w{width_cm}_h{height_cm}.urdf"
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    W = width_cm / 100.0
    H = height_cm / 100.0
    T = 0.0049  # Kalınlığı yuvarladık (10mm)

    half_W = W / 2.0
    half_H = H / 2.0
    half_T = T / 2.0
    wall_h = H - T
    
    bottom_z = -half_H + half_T
    sides_z = half_T
    offset = half_W - half_T
    inner_len = W - (2 * T)

    # --- FİZİKSEL İYİLEŞTİRMELER ---
    mass = 0.2  # Kütleyi biraz artırdık
    # Daha dengeli bir kule için kütle merkezini tabana (bottom_z) çektik
    center_of_mass_z = bottom_z 
    
    # Atalet momenti hesaplaması (Kutu formülü iyileştirildi)
    ixx = (1/12.0) * mass * (W**2 + H**2)
    iyy = (1/12.0) * mass * (W**2 + H**2)
    izz = (1/12.0) * mass * (W**2 + W**2)

    urdf_content = f"""<?xml version="1.0"?>
<robot name="cup_{width_cm}x{height_cm}cm">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 {center_of_mass_z:.4f}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>

    <visual>
      <origin xyz="0 0 {bottom_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {W:.4f} {T:.4f}"/></geometry>
      <material name="green"><color rgba="0 1 0 0.8"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {bottom_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {W:.4f} {T:.4f}"/></geometry>
    </collision>

    <visual>
      <origin xyz="0 {offset:.4f} {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {T:.4f} {wall_h:.4f}"/></geometry>
      <material name="blue"><color rgba="0 0.5 1 0.5"/></material>
    </visual>
    <collision>
      <origin xyz="0 {offset:.4f} {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {T:.4f} {wall_h:.4f}"/></geometry>
    </collision>

    <visual>
      <origin xyz="0 -{offset:.4f} {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {T:.4f} {wall_h:.4f}"/></geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 -{offset:.4f} {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {T:.4f} {wall_h:.4f}"/></geometry>
    </collision>

    <visual>
      <origin xyz="{offset:.4f} 0 {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{T:.4f} {inner_len:.4f} {wall_h:.4f}"/></geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="{offset:.4f} 0 {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{T:.4f} {inner_len:.4f} {wall_h:.4f}"/></geometry>
    </collision>

    <visual>
      <origin xyz="-{offset:.4f} 0 {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{T:.4f} {inner_len:.4f} {wall_h:.4f}"/></geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="-{offset:.4f} 0 {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{T:.4f} {inner_len:.4f} {wall_h:.4f}"/></geometry>
    </collision>
  </link>
</robot>
"""
    with open(filename, "w") as f:
        f.write(urdf_content)
    print(f"Cup generated with Inertia: {filename} ({width_cm}x{height_cm}cm)")

# Örnek Kullanım:
generate_cup_urdf(16, 16)
generate_cup_urdf(18, 18)
generate_cup_urdf(20, 20)