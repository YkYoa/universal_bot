import mujoco
import mujoco.viewer
import time
import rclpy
from rclpy.node import Node
import os

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        # Use the MJCF file we converted
        model_path = "/home/hans/universal_bot/Open_arm_a1_ws/src/openarm_mujoco_sim/config/openarm.xml/mujoco_description.xml"
        
        self.get_logger().info(f"Loading model from: {model_path}")
        
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Launch the passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Create a timer for the physics step
        self.timer = self.create_timer(0.01, self.step)

    def step(self):
        # Physics step
        mujoco.mj_step(self.model, self.data)
        # Sync the viewer
        if self.viewer.is_running():
            self.viewer.sync()
        else:
            self.get_logger().info("Viewer closed")
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
