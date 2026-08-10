#!/usr/bin/env python3
import os
import threading
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

# Flask & Socket.IO
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory telemetry - sequence_executor publishes whole-sequence status
# (not per-node transitions like BT.CPP did), so node_states here is coarse:
# every node in the tree shares the sequence's own current status.
overall_status = 'IDLE'
status_log = []


def build_sequence_tree(sequence_yaml_path, sequence_name):
    """Builds the same {id,name,type,attributes,status,children} shape the
    frontend already renders (previously produced by parsing BT XML) -
    directly from a `sequences:` entry in sequence.yaml instead. Reuses BT
    node type strings (PlanToJointTarget/PlanToJointSequence/Parallel/
    SetHandYaw/SetHandFlex/Repeat/Sequence) purely so the frontend's
    existing shape/color classification keeps working unchanged - this
    tree is NOT executed, it's just a visual approximation of what
    sequence_executor's fixed-shape interpreter actually does (home once,
    then loop the body)."""
    with open(sequence_yaml_path) as f:
        data = yaml.safe_load(f) or {}

    sequences = data.get('sequences', {})
    if sequence_name not in sequences:
        return None
    seq = sequences[sequence_name] or {}

    counter = [0]

    def new_node(name, ntype, children=None):
        counter[0] += 1
        return {
            'id': f'node_{counter[0]}',
            'name': name,
            'type': ntype,
            'attributes': {},
            'status': overall_status,
            'children': children or [],
        }

    root_children = []

    home_section = seq.get('home_section')
    if home_section:
        home_children = [new_node(f'Home ({home_section})', 'PlanToJointTarget')]
        section = data.get(home_section, {}) or {}
        has_hand = any(str(k).startswith('lh') or str(k).startswith('rh') for k in section.keys())
        if has_hand:
            home_children.append(new_node('Hand Hold', 'Parallel', [
                new_node('SetHandYaw', 'SetHandYaw'),
                new_node('SetHandFlex', 'SetHandFlex'),
            ]))
        root_children.append(new_node('Home (once)', 'Sequence', home_children))

    body_children = []
    if seq.get('body_sections'):
        names = [s.strip() for s in str(seq['body_sections']).split(',') if s.strip()]
        for name in names:
            body_children.append(new_node(f'Hand pose: {name}', 'SetHandFlex'))
    elif seq.get('body_section'):
        label = seq['body_section']
        if seq.get('body_right_section'):
            label += f" + {seq['body_right_section']}"
        body_children.append(new_node(f'Wave: {label}', 'PlanToJointSequence'))

    repeat = seq.get('repeat', 1)
    if repeat != 1:
        times = 'forever' if repeat == -1 else f'{repeat}x'
        root_children.append(new_node(f'Body (repeat {times})', 'Repeat', body_children))
    else:
        root_children.extend(body_children)

    return new_node(sequence_name, 'Sequence', root_children)


class BTViewerNode(Node):
    def __init__(self):
        super().__init__('bt_viewer_node')

        self.declare_parameter('sequence_yaml_path', '')
        self.declare_parameter('sequence_name', '')
        self.declare_parameter('port', 5000)

        app.config['SEQUENCE_YAML_PATH'] = self.get_parameter('sequence_yaml_path').value
        app.config['SEQUENCE_NAME'] = self.get_parameter('sequence_name').value

        self.get_logger().info(
            f"Watching sequence '{app.config['SEQUENCE_NAME']}' in {app.config['SEQUENCE_YAML_PATH']}")

        # sequence_executor's node name is "sequence_executor" (set in
        # sequence_executor.launch.py), so its ~/status and ~/fault topics
        # resolve to /sequence_executor/status and /sequence_executor/fault.
        self.status_sub = self.create_subscription(
            String, '/sequence_executor/status', self.status_callback, 10)
        self.fault_sub = self.create_subscription(
            String, '/sequence_executor/fault', self.fault_callback, 10)

        self.get_logger().info("BT Viewer Node subscriptions started")

    def status_callback(self, msg):
        global overall_status
        # sequence_executor publishes "<sequence_name>: <status text>" -
        # map a few recognizable phrases to the RUNNING/SUCCESS vocabulary
        # the frontend already understands; anything else just passes
        # through as informational text under the RUNNING color.
        text = msg.data
        if text.endswith(': done'):
            overall_status = 'SUCCESS'
        else:
            overall_status = 'RUNNING'
        status_log.append({'text': text, 'level': 'status'})
        if len(status_log) > 100:
            status_log.pop(0)
        socketio.emit('tree_status', {'status': overall_status, 'text': text})
        socketio.emit('log_line', {'text': text, 'level': 'status'})

    def fault_callback(self, msg):
        global overall_status
        overall_status = 'FAILURE'
        status_log.append({'text': msg.data, 'level': 'fault'})
        if len(status_log) > 100:
            status_log.pop(0)
        socketio.emit('tree_status', {'status': overall_status, 'text': msg.data})
        socketio.emit('log_line', {'text': msg.data, 'level': 'fault'})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tree')
def get_tree():
    yaml_path = app.config.get('SEQUENCE_YAML_PATH')
    sequence_name = app.config.get('SEQUENCE_NAME')
    if not yaml_path or not os.path.exists(yaml_path):
        return jsonify({"error": f"sequence.yaml not found at {yaml_path}"}), 404
    if not sequence_name:
        return jsonify({"error": "sequence_name param not set"}), 404
    try:
        tree = build_sequence_tree(yaml_path, sequence_name)
        if tree is None:
            return jsonify({"error": f"sequences:{sequence_name} not found in {yaml_path}"}), 404
        return jsonify(tree)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status')
def get_status():
    return jsonify({"status": overall_status})


@app.route('/api/logs')
def get_logs():
    return jsonify(status_log)


def main(args=None):
    rclpy.init(args=args)
    global ros_node
    ros_node = BTViewerNode()

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
