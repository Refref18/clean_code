import os

def generate_inverted_cup_urdf(width_cm, height_cm, filename=None):
    """
    Generates an inverted cup URDF (bottom plate on top).
    Width and Depth = width_cm.
    Height = height_cm.
    """
    if filename is None:
        filename = f"./objects_new/inv_cup_w{width_cm}_h{height_cm}_thinner.urdf"
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert cm to meters
    W = width_cm / 100.0
    H = height_cm / 100.0
    T = 0.0049  # Thickness (~0.45cm)
    
    # Half dimensions for calculations
    half_W = W / 2.0
    half_H = H / 2.0
    half_T = T / 2.0
    
    # Height of the side walls (Total height minus plate thickness)
    wall_h = H - T
    
    # Z-offsets for Inverted logic (centered at 0,0,0)
    # Plate is at the top (+Z)
    plate_z = half_H - half_T
    # Walls center shifted downward (-Z)
    sides_z = -half_T
    
    # X/Y Offsets for the walls
    offset = half_W - half_T
    inner_len = W - (2 * T)
    
    # Yellow marker placement
    # Marker size scales with height; marker_y_offset sits just outside the wall
    marker_size = 0.04 * (height_cm / 10.0)
    marker_y_offset = half_W + 0.0005 

    urdf_content = f"""<?xml version="1.0"?>
<robot name="inverted_cup_{width_cm}x{height_cm}cm">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 {plate_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {W:.4f} {T:.4f}"/></geometry>
      <material name="red"><color rgba="1 0 0 0.7"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {plate_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {W:.4f} {T:.4f}"/></geometry>
    </collision>

    <visual>
      <origin xyz="0 {offset:.4f} {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {T:.4f} {wall_h:.4f}"/></geometry>
      <material name="blue"><color rgba="0 0 1 0.4"/></material>
    </visual>
    <collision>
      <origin xyz="0 {offset:.4f} {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{W:.4f} {T:.4f} {wall_h:.4f}"/></geometry>
    </collision>

    <visual>
      <origin xyz="0 {marker_y_offset:.5f} {sides_z:.4f}" rpy="0 0 0"/>
      <geometry><box size="{marker_size:.4f} 0.001 {marker_size:.4f}"/></geometry>
      <material name="yellow"><color rgba="1 1 0 1"/></material>
    </visual>

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
    print(f"Inverted Cup generated: {filename} (W:{width_cm}cm, H:{height_cm}cm)")


# Example Usage:
generate_inverted_cup_urdf(16, 16)
generate_inverted_cup_urdf(18, 18)
generate_inverted_cup_urdf(20, 20)