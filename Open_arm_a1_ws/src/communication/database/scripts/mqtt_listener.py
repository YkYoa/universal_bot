#!/usr/bin/env python3
import os
import sys
import json
import time
import urllib.parse
import threading
from datetime import datetime, timezone
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import paho.mqtt.client as mqtt

class MqttListenerNode(Node):
    def __init__(self):
        super().__init__('mqtt_listener_node')
        
        # Read parameters / env variables
        self.robot_id = os.environ.get("ROBOT_ID", "aaabbbccc")
        self.get_logger().info(f"Robot ID configured: {self.robot_id}")
        
        broker_url_str = os.environ.get("MQTT_BROKER_URL", "mqtt://localhost:1883")
        self.get_logger().info(f"MQTT Broker URL configured: {broker_url_str}")
        
        # Parse broker URL
        host = "localhost"
        port = 1883
        use_tls = False
        if not (broker_url_str.startswith("mqtt://") or broker_url_str.startswith("mqtts://") or 
                broker_url_str.startswith("tcp://") or broker_url_str.startswith("ssl://")):
            broker_url_str = "mqtt://" + broker_url_str
            
        parsed = urllib.parse.urlparse(broker_url_str)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port:
            port = parsed.port
        if parsed.scheme in ("mqtts", "ssl"):
            use_tls = True
            
        self.mqtt_user = os.environ.get("MQTT_USER", None)
        self.mqtt_pass = os.environ.get("MQTT_PASS", None)
        
        self.ack_endpoint = os.environ.get(
            "ACK_ENDPOINT", 
            f"{os.environ.get('BACKEND_URL', 'http://localhost:8080')}/api/robot-commands/ack"
        )
        self.get_logger().info(f"HTTP ACK Endpoint: {self.ack_endpoint}")
        
        # ROS 2 Publisher & Subscriber
        self.command_pub = self.create_publisher(String, '/openarm_demo/command', 10)
        self.status_sub = self.create_subscription(String, '/openarm_demo/status', self.status_callback, 10)
        
        # Thread safety lock and states
        self.lock = threading.Lock()
        self.robot_status = "idle"         # "idle" or "executing"
        
        self.active_intent = None
        self.active_correlation_id = None
        self.active_command_started = False
        self.active_command_timestamp = None
        
        # Timeout timer (checks active command timeout)
        self.timeout_timer = self.create_timer(1.0, self.check_timeout_callback)
        
        # Initialize MQTT client
        try:
            from paho.mqtt import callback_api_version
            self.mqtt_client = mqtt.Client(callback_api_version.CallbackAPIVersion.VERSION1, client_id=f"robot_listener_{self.robot_id}")
        except Exception:
            self.mqtt_client = mqtt.Client(client_id=f"robot_listener_{self.robot_id}")
            
        if self.mqtt_user and self.mqtt_pass:
            self.mqtt_client.username_pw_set(self.mqtt_user, self.mqtt_pass)
            
        if use_tls:
            # Set rejectUnauthorized: false compatibility by setting insecure TLS
            self.mqtt_client.tls_set()
            self.mqtt_client.tls_insecure_set(True)
            
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        
        # Connect to MQTT
        self.get_logger().info(f"Connecting to MQTT Broker {host}:{port} ...")
        try:
            self.mqtt_client.connect(host, port, keepalive=60)
            self.mqtt_client.loop_start()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to MQTT Broker: {e}")

    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            topic = f"robots/{self.robot_id}/commands"
            self.get_logger().info(f"Connected to MQTT Broker! Subscribing to topic: {topic}")
            self.mqtt_client.subscribe(topic, qos=1)
        else:
            self.get_logger().error(f"MQTT connection failed with code {rc}")

    def on_mqtt_disconnect(self, client, userdata, rc):
        self.get_logger().warn(f"Disconnected from MQTT Broker with code {rc}. Reconnecting...")

    def send_http_ack(self, correlation_id, intent, status, success, error_msg=None):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "robotId": self.robot_id,
            "correlationId": correlation_id,
            "intent": intent,
            "status": status,
            "success": success,
            "error": error_msg
        }
        self.get_logger().info(f"Sending HTTP ACK: {json.dumps(payload)}")
        try:
            response = requests.post(self.ack_endpoint, json=payload, timeout=5.0)
            self.get_logger().info(f"HTTP ACK Response Status: {response.status_code}")
        except Exception as e:
            self.get_logger().error(f"Failed to send HTTP ACK to backend: {e}")

    def on_mqtt_message(self, client, userdata, msg):
        try:
            raw_payload = msg.payload.decode('utf-8')
            self.get_logger().info(f"Received MQTT message on topic: {msg.topic}")
            
            try:
                data = json.loads(raw_payload)
            except json.JSONDecodeError:
                self.get_logger().error(f"Payload is not valid JSON: {raw_payload}")
                return
                
            robot_id = data.get("robotId")
            timestamp = data.get("timestamp")
            intent = data.get("intent")
            success = data.get("success")
            
            # Validate robotId equals local robot id
            if robot_id != self.robot_id:
                self.get_logger().warn(f"Ignored command for robotId '{robot_id}'. Expected '{self.robot_id}'.")
                return
                
            # Validate timestamp, intent, and success exist
            if not timestamp or not isinstance(intent, str) or success is None:
                self.get_logger().error("Invalid payload contract: missing timestamp, intent, or success fields.")
                return
                
            # If success !== true, ignore the command and log it
            if success is not True:
                self.get_logger().warn(f"Ignored command because success field is not true: {raw_payload}")
                return
                
            # Retrieve correlationId from payload
            inner_payload = data.get("payload", {})
            correlation_id = inner_payload.get("correlationId") if isinstance(inner_payload, dict) else None
            if not correlation_id:
                correlation_id = f"gen-{int(time.time())}"
                
            self.get_logger().info(f"Processing command intent: '{intent}' (correlationId: {correlation_id})")
            
            # Send HTTP ACK 'received'
            self.send_http_ack(correlation_id, intent, "received", True)
            
            # Route by intent: only greet and wave poses are focused for database team
            if intent not in ("greet", "wave"):
                err_msg = f"Unsupported intent: '{intent}'."
                self.get_logger().warn(f"[MQTT] {err_msg}")
                self.send_http_ack(correlation_id, intent, "failed", False, err_msg)
                return
                
            # Enforce local safety check before movement
            with self.lock:
                if self.robot_status == "executing" or self.active_correlation_id is not None:
                    err_msg = "Robot is busy executing another command."
                    self.get_logger().warn(f"[Safety Check] {err_msg}")
                    self.send_http_ack(correlation_id, intent, "failed", False, err_msg)
                    return
                    
                # Mark as starting
                self.active_intent = intent
                self.active_correlation_id = correlation_id
                self.active_command_started = False
                self.active_command_timestamp = time.time()
                
            # Send HTTP ACK 'started'
            self.send_http_ack(correlation_id, intent, "started", True)
            
            # Publish to demo node command topic
            cmd_msg = String()
            if intent == "greet":
                cmd_msg.data = "gc"  # Cartesian greeting sequence
            else:
                cmd_msg.data = "wc"  # Cartesian waving sequence
                
            self.get_logger().info(f"Publishing command '{cmd_msg.data}' to /openarm_demo/command")
            self.command_pub.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error handling MQTT message: {e}")

    def status_callback(self, msg):
        status_data = msg.data.strip()
        self.get_logger().info(f"Received status update from demo node: '{status_data}'")
        
        send_ack = False
        corr_id = None
        intent = None
        
        with self.lock:
            if status_data.startswith("executing:"):
                self.robot_status = "executing"
                if self.active_correlation_id:
                    self.active_command_started = True
            elif status_data == "idle":
                was_executing = (self.robot_status == "executing")
                self.robot_status = "idle"
                
                if self.active_correlation_id and (self.active_command_started or was_executing):
                    corr_id = self.active_correlation_id
                    intent = self.active_intent
                    send_ack = True
                    
                    # Clear active command
                    self.active_correlation_id = None
                    self.active_intent = None
                    self.active_command_started = False
                    self.active_command_timestamp = None
                    
        if send_ack:
            self.send_http_ack(corr_id, intent, "completed", True)

    def check_timeout_callback(self):
        send_timeout = False
        corr_id = None
        intent = None
        
        with self.lock:
            if self.active_correlation_id and self.active_command_timestamp:
                elapsed = time.time() - self.active_command_timestamp
                # If command is active and has been running for more than 45 seconds, time it out
                if elapsed > 45.0:
                    corr_id = self.active_correlation_id
                    intent = self.active_intent
                    send_timeout = True
                    
                    # Clear active command
                    self.active_correlation_id = None
                    self.active_intent = None
                    self.active_command_started = False
                    self.active_command_timestamp = None
                    self.robot_status = "idle"
                    
        if send_timeout:
            err_msg = "Command execution timed out after 45 seconds."
            self.get_logger().error(err_msg)
            self.send_http_ack(corr_id, intent, "timed_out", False, err_msg)

    def destroy_node(self):
        self.get_logger().info("Stopping MQTT client loop...")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MqttListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
