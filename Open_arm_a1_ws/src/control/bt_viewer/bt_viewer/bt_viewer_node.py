#!/usr/bin/env python3
import os
import sys
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory

# Flask & Socket.IO
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

# XML parser
import xml.etree.ElementTree as ET

# We create the Flask application
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory storage for status updates
blackboard_state = {}
node_states = {}
transition_logs = []

class BTViewerNode(Node):
    def __init__(self):
        super().__init__('bt_viewer_node')
        
        # Parameters
        self.declare_parameter('bt_xml_path', '')
        self.declare_parameter('port', 5000)
        
        # Determine the BT XML Path
        xml_path = self.get_parameter('bt_xml_path').value
        if not xml_path:
            try:
                xml_path = os.path.join(
                    get_package_share_directory('bt_executor'),
                    'bt_trees', 'pick_and_place.xml'
                )
            except Exception as e:
                self.get_logger().error(f"Could not find bt_executor package: {e}")
                xml_path = ""
                
        self.get_logger().info(f"Using Behavior Tree XML: {xml_path}")
        app.config['BT_XML_PATH'] = xml_path
        
        # Subscribers
        self.status_sub = self.create_subscription(
            String,
            '/bt_executor/status',
            self.status_callback,
            10
        )
        self.blackboard_sub = self.create_subscription(
            String,
            '/bt_executor/blackboard',
            self.blackboard_callback,
            10
        )
        self.transitions_sub = self.create_subscription(
            String,
            '/bt_executor/transitions',
            self.transitions_callback,
            10
        )
        
        # Service clients
        self.replan_client = self.create_client(Trigger, '/bt_executor/replan')
        self.goal_updated_client = self.create_client(Trigger, '/bt_executor/goal_updated')
        
        # Publishers (for testing/triggering)
        self.grasp_pose_pub = self.create_publisher(
            PoseStamped,
            '/bt_executor/vla_grasp_pose',
            10
        )
        
        self.get_logger().info("BT Viewer Node subscriptions started")

    def status_callback(self, msg):
        socketio.emit('tree_status', {'status': msg.data})

    def blackboard_callback(self, msg):
        try:
            data = json.loads(msg.data)
            global blackboard_state
            blackboard_state.update(data)
            socketio.emit('blackboard_update', data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse blackboard json: {e}")

    def transitions_callback(self, msg):
        try:
            data = json.loads(msg.data)
            global node_states, transition_logs
            node_states[data['node_name']] = data['status']
            transition_logs.append(data)
            if len(transition_logs) > 100:
                transition_logs.pop(0)
            socketio.emit('transition_update', data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse transition json: {e}")

    def call_replan(self):
        if not self.replan_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Replan service not available")
            return False, "Replan service not available"
        req = Trigger.Request()
        self.replan_client.call_async(req)
        return True, "Replan triggered"

    def call_goal_updated(self):
        if not self.goal_updated_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Goal updated service not available")
            return False, "Goal updated service not available"
        req = Trigger.Request()
        self.goal_updated_client.call_async(req)
        return True, "Goal updated triggered"

    def publish_grasp_pose(self, x, y, z):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.grasp_pose_pub.publish(msg)
        self.get_logger().info(f"Published custom grasp pose: {x}, {y}, {z}")
        return True

# Flask Web Server Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tree')
def get_tree():
    xml_path = app.config.get('BT_XML_PATH')
    if not xml_path or not os.path.exists(xml_path):
        return jsonify({"error": f"XML file {xml_path} not found"}), 404
    
    try:
        structure = parse_bt_xml(xml_path)
        return jsonify(structure)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/blackboard')
def get_blackboard():
    return jsonify(blackboard_state)

@app.route('/api/transitions')
def get_transitions():
    return jsonify(transition_logs)

@app.route('/api/node_states')
def get_node_states():
    return jsonify(node_states)

@app.route('/api/replan', methods=['POST'])
def trigger_replan():
    success, msg = ros_node.call_replan()
    return jsonify({"success": success, "message": msg})

@app.route('/api/goal_updated', methods=['POST'])
def trigger_goal_updated():
    success, msg = ros_node.call_goal_updated()
    return jsonify({"success": success, "message": msg})

@app.route('/api/publish_pose', methods=['POST'])
def publish_pose():
    try:
        data = request.json
        x = data.get('x', 0.0)
        y = data.get('y', 0.0)
        z = data.get('z', 0.0)
        ros_node.publish_grasp_pose(x, y, z)
        return jsonify({"success": True, "message": f"Published pose ({x}, {y}, {z})"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

def parse_bt_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        main_tree_id = root.attrib.get('main_tree_to_execute')
        bt_element = None
        for child in root:
            if child.tag == 'BehaviorTree' and child.attrib.get('ID') == main_tree_id:
                bt_element = child
                break
        
        if bt_element is None:
            for child in root:
                if child.tag == 'BehaviorTree':
                    bt_element = child
                    break
                    
        if bt_element is None:
            return None
            
        counter = [0]
        
        def traverse(elem):
            node_type = elem.tag
            attrs = dict(elem.attrib)
            
            node_id = f"node_{counter[0]}"
            counter[0] += 1
            
            node_name = attrs.get('name', node_type)
            
            node_data = {
                'id': node_id,
                'name': node_name,
                'type': node_type,
                'attributes': attrs,
                'status': node_states.get(node_name, 'IDLE'),
                'children': []
            }
            
            for child in elem:
                child_node = traverse(child)
                if child_node:
                    node_data['children'].append(child_node)
                    
            return node_data
            
        if len(bt_element) > 0:
            return traverse(bt_element[0])
        return None
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

def main(args=None):
    rclpy.init(args=args)
    global ros_node
    ros_node = BTViewerNode()
    
    # Run rclpy spin in background thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()
    
    port = ros_node.get_parameter('port').value
    ros_node.get_logger().info(f"Starting Web server on port {port}")
    
    try:
        socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
