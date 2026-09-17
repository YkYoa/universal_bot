"""ros2_control controller-exclusivity switch: the mechanism behind
interlock 1 (plan). Engaging gesture teleop for one arm deactivates that
arm's `*_arm_controller` (JointTrajectoryController - what MoveIt/the FSM/
the Android app command through) and activates its
`*_forward_position_controller` (ForwardCommandController - what
command_sink.ForwardPositionSink streams to). This makes "MoveIt cannot
move an arm gesture-teleop currently holds" an OS-level fact enforced by
ros2_control itself, not a convention two independent code paths have to
both remember to respect.

`*_forward_position_controller` is declared in bimanual_controllers.yaml
but NOT auto-spawned (see that file's own comment, lines ~26-43) - it must
be `load_controller`'d (to `inactive`) before the first switch, once per
controller_manager lifetime; ros2_control errors loudly (not silently) if
you switch_controller to a name that was never loaded, so callers should
treat a load failure as fatal rather than proceeding to switch anyway.

Follows fsm_bridge.py's non-spinning `_wait()` pattern exactly: this
class's node is spun by teleop_node's own executor thread (mirroring
robot_api_server.py's setup), so spinning again from wherever engage()/
disengage() is called would re-enter the executor and deadlock - waiting
on the future's own completion event is the safe half.
"""
from __future__ import annotations

import threading

from controller_manager_msgs.srv import LoadController, SwitchController

SERVICE_TIMEOUT_S = 5.0
SWITCH_TIMEOUT_S = 2.0  # ros2_control's own controller-switch deadline, not the service-call wait above

ARM_CONTROLLER = {"left": "left_arm_controller", "right": "right_arm_controller"}
FORWARD_POSITION_CONTROLLER = {"left": "left_forward_position_controller", "right": "right_forward_position_controller"}

# controller_manager_msgs/srv/SwitchController.Request.strictness values
# (STRICT: the whole switch fails if any one controller can't be
# start/stopped, rather than partially succeeding - the only sane choice
# for an exclusivity switch).
STRICT = 2


def _wait(future, timeout_s):
    """See fsm_bridge.py's identical helper - do not spin here."""
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout_s) and future.done()


class ControllerSwitch:
    def __init__(self, node):
        self._node = node
        self._load_client = node.create_client(LoadController, "/controller_manager/load_controller")
        self._switch_client = node.create_client(SwitchController, "/controller_manager/switch_controller")
        self._loaded = set()

    def ensure_loaded(self, controller_name: str):
        """Idempotent: load_controller on an already-loaded controller is a
        harmless no-op per ros2_control, but this still tracks locally to
        avoid an unnecessary service round-trip on every engage()."""
        if controller_name in self._loaded:
            return True, "already loaded"
        if not self._load_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            return False, "/controller_manager/load_controller is not available"
        request = LoadController.Request()
        request.name = controller_name
        future = self._load_client.call_async(request)
        if not _wait(future, SERVICE_TIMEOUT_S):
            return False, f"load_controller('{controller_name}') timed out"
        response = future.result()
        if response.ok:
            self._loaded.add(controller_name)
        return response.ok, "" if response.ok else f"load_controller('{controller_name}') failed"

    def _switch(self, start: list, stop: list):
        if not self._switch_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            return False, "/controller_manager/switch_controller is not available"
        request = SwitchController.Request()
        request.start_controllers = start
        request.stop_controllers = stop
        request.strictness = STRICT
        request.activate_asap = True
        request.timeout.sec = int(SWITCH_TIMEOUT_S)
        future = self._switch_client.call_async(request)
        if not _wait(future, SERVICE_TIMEOUT_S):
            return False, "switch_controller timed out"
        response = future.result()
        return response.ok, "" if response.ok else "switch_controller reported failure"

    def engage(self, side: str):
        """Deactivate `<side>_arm_controller`, activate
        `<side>_forward_position_controller`. Returns (success, reason)."""
        forward = FORWARD_POSITION_CONTROLLER[side]
        ok, reason = self.ensure_loaded(forward)
        if not ok:
            return False, reason
        return self._switch(start=[forward], stop=[ARM_CONTROLLER[side]])

    def disengage(self, side: str):
        """Reverse of engage(): restores `<side>_arm_controller` so
        MoveIt/the FSM/the Android app can command the arm again."""
        forward = FORWARD_POSITION_CONTROLLER[side]
        return self._switch(start=[ARM_CONTROLLER[side]], stop=[forward])
