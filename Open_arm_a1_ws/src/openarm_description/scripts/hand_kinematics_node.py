#!/usr/bin/python3
"""Solves the amazing_hand's closed-loop rod/gimbal linkage.

The exported URDF (from Onshape) represents each finger's parallel rod
mechanism as an open tree: two rod-driven ball joints and a proximal/distal
knuckle got cut to break the real kinematic loops, leaving disconnected
passive joints that a plain joint_state_publisher can't move correctly.

This node reads the URDF once at startup to discover, per finger, the two
driving servo joints and the geometry of its rod/gimbal/knuckle linkage
(rod lengths and knuckle offset calibrated from the assembled rest pose),
then on every joint_states_raw update solves each finger's small nonlinear
system (least squares) and republishes a complete, correct joint state.

Two command spaces (parameter `command_space`), mirroring the FK/IK pair in
tomo_control's tomo_finger_kinematic.cpp (convertPrismaticToAngles /
convertAnglesToPrismatics):

- 'servo':   input = the 8 servo horn angles; passive linkage joints are
             solved forward from them.
- 'knuckle': input = per-finger knuckle angles -- yaw/abduction (the
             bushing->gimbal revolute) and proximal flexion (the
             gimbal->proximal revolute); the node solves the inverse
             problem for both servo angles plus the remaining passive
             joints, so the published joint state also tells you what to
             command the real SCS0009s. The distal knuckle is mechanically
             coupled to proximal flexion through the linkage (2 servos =
             2 DOF per finger) and follows automatically.
"""
import re
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.optimize import least_squares
from sensor_msgs.msg import JointState


def rpy_to_matrix(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_rotation(axis, theta):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


class Kinematics:
    """Generic URDF forward-kinematics helper (root-frame transforms)."""

    def __init__(self, urdf_xml):
        root = ET.fromstring(urdf_xml)
        self.joints = {}
        self.children = {}
        self.parent_of_link = {}
        self.all_links = {l.get('name') for l in root.findall('link')}
        for j in root.findall('joint'):
            name = j.get('name')
            o = j.find('origin')
            a = j.find('axis')
            lim = j.find('limit')
            xyz = np.array([float(v) for v in (o.get('xyz') if o is not None else '0 0 0').split()])
            rpy = np.array([float(v) for v in (o.get('rpy') if o is not None else '0 0 0').split()])
            axis = np.array([float(v) for v in (a.get('xyz') if a is not None else '0 0 1').split()])
            p = j.find('parent').get('link')
            c = j.find('child').get('link')
            info = dict(name=name, type=j.get('type'), xyz=xyz, rpy=rpy, axis=axis,
                        parent=p, child=c,
                        lower=float(lim.get('lower')) if lim is not None else None,
                        upper=float(lim.get('upper')) if lim is not None else None)
            self.joints[name] = info
            self.children.setdefault(p, []).append(name)
            self.parent_of_link[c] = name

    def joint_local_T(self, jname, value):
        j = self.joints[jname]
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*j['rpy'])
        T[:3, 3] = j['xyz']
        if j['type'] in ('revolute', 'continuous'):
            Tm = np.eye(4)
            Tm[:3, :3] = axis_rotation(j['axis'], value)
            return T @ Tm
        elif j['type'] == 'prismatic':
            Tm = np.eye(4)
            Tm[:3, 3] = j['axis'] / np.linalg.norm(j['axis']) * value
            return T @ Tm
        return T

    def fk(self, link, values=None):
        values = values or {}
        chain = []
        cur = link
        while cur in self.parent_of_link:
            jn = self.parent_of_link[cur]
            chain.append(jn)
            cur = self.joints[jn]['parent']
        T = np.eye(4)
        for jn in reversed(chain):
            T = T @ self.joint_local_T(jn, values.get(jn, 0.0))
        return T

    def child_links(self, link, jtype=None):
        out = []
        for jn in self.children.get(link, []):
            j = self.joints[jn]
            if jtype is None or j['type'] == jtype:
                out.append((jn, j['child']))
        return out

    def ancestors(self, link):
        path = []
        cur = link
        while cur in self.parent_of_link:
            jn = self.parent_of_link[cur]
            cur = self.joints[jn]['parent']
            path.append(cur)
        return path


def discover_fingers(hand, link_prefix=''):
    """Find each finger's gimbal/servo/rod-ball/knuckle joints and links.

    See the amazing_hand mechanism notes in this file's module docstring.
    Discovery is purely structural (joint types + tree ancestry), not
    dependent on any particular Onshape instance-numbering convention,
    since that numbering is inconsistent across the 4 fingers as exported.

    link_prefix restricts discovery to one hand's links when the combined
    robot_description contains both (e.g. 'openarm_left_ahand_'); the two
    hands' subtrees are disjoint (different mount points) so this only
    needs to scope the servo_links/bushing_links seed patterns -- ancestry
    walks never cross between them regardless.
    """
    p = re.escape(link_prefix)
    servo_links = sorted(l for l in hand.all_links if re.fullmatch(rf'{p}scs0009(_\d+)?', l))
    bushing_links = sorted(l for l in hand.all_links if re.fullmatch(rf'{p}bushing_0608_04(_\d+)?', l))

    gimbals = [(jn, j['parent'], j['child']) for jn, j in hand.joints.items()
               if j['type'] == 'revolute' and j['parent'] in bushing_links]

    fingers = []
    for q1_joint, bushing, gimbal in gimbals:
        host, servos = None, None
        for cand in hand.ancestors(bushing):
            servos_here = [s for s in servo_links if cand in hand.ancestors(s)]
            if len(servos_here) == 2:
                host, servos = cand, servos_here
                break
        if host is None:
            raise RuntimeError(f"could not find a 2-servo host ancestor for gimbal {gimbal}")

        kids = hand.child_links(gimbal)
        proximal_j = proximal_link = link_body_j = link_body = None
        for kjn, kc in kids:
            kj = hand.joints[kjn]
            if kj['type'] == 'fixed':
                continue
            if kj['type'] == 'revolute' and kj['lower'] is not None:
                proximal_j, proximal_link = kjn, kc
            elif kj['type'] == 'continuous':
                link_body_j, link_body = kjn, kc
        if not proximal_link or not link_body:
            raise RuntimeError(f"gimbal {gimbal}: expected a proximal + link_body child pair")

        rb_children = [c for _, c in hand.child_links(link_body, 'fixed') if 'rotule_ball' in c]
        if len(rb_children) != 2:
            raise RuntimeError(f"link_body {link_body}: expected 2 rotule_ball children, got {rb_children}")

        prisms = hand.child_links(link_body, 'prismatic')
        if len(prisms) != 1:
            raise RuntimeError(f"link_body {link_body}: expected 1 prismatic child")
        prism_j, cyl = prisms[0]
        conts = hand.child_links(cyl, 'continuous')
        if len(conts) != 1:
            raise RuntimeError(f"{cyl}: expected 1 continuous child (distal twist)")
        twist_j, distal_real = conts[0]

        pkids = hand.child_links(proximal_link, 'continuous')
        if len(pkids) != 1:
            raise RuntimeError(f"proximal {proximal_link}: expected 1 continuous knuckle child")
        knuckle_j, distal_dup = pkids[0]

        servo_data = []
        for sv in servos:
            acts = [(jn, j) for jn, j in hand.joints.items()
                    if j['parent'] == sv and j['type'] == 'revolute' and j['lower'] is not None]
            if len(acts) != 1:
                raise RuntimeError(f"servo {sv}: expected 1 actuator joint, got {acts}")
            act_jn, act_j = acts[0]
            horn = act_j['child']
            hb = [c for _, c in hand.child_links(horn, 'fixed') if 'rotule_ball' in c]
            if len(hb) != 1:
                raise RuntimeError(f"horn {horn}: expected 1 rotule_ball child")
            servo_data.append(dict(servo=sv, actuator_joint=act_jn, horn=horn, horn_ball=hb[0],
                                    lower=act_j['lower'], upper=act_j['upper']))

        fingers.append(dict(
            gimbal=gimbal, q1_joint=q1_joint, bushing=bushing, host=host,
            q2_joint=link_body_j, link_body=link_body,
            prox_joint=proximal_j, proximal=proximal_link,
            prox_lower=hand.joints[proximal_j]['lower'], prox_upper=hand.joints[proximal_j]['upper'],
            q1_lower=hand.joints[q1_joint]['lower'], q1_upper=hand.joints[q1_joint]['upper'],
            knuckle_joint=knuckle_j, distal_dup=distal_dup,
            prism_joint=prism_j, twist_joint=twist_j, distal_real=distal_real,
            rod_balls=rb_children, servo_data=servo_data,
        ))
    return fingers


def ball_chain(hand, start_link):
    """Follow the decomposed spherical joint (3 stacked continuous joints)
    hanging off a rotule_ball link; returns (joint_names, terminal_link)."""
    joints, cur = [], start_link
    while True:
        kids = hand.child_links(cur, 'continuous')
        if not kids:
            return joints, cur
        jn, c = kids[0]
        joints.append(jn)
        cur = c


def calibrate(hand, fingers):
    """Compute rod lengths and knuckle offset from the assembled rest pose.

    Pairs each link-side rod ball with its servo by rest-pose distance
    (bijective, minimum total): each physical rod connects the closest
    ball pair, and getting this wrong swaps which servo drives which side
    of the gimbal. List order out of discovery is arbitrary (and set
    iteration order even varies between runs), so never trust it.
    """
    rest_pos = {link: hand.fk(link)[:3, 3] for link in hand.all_links}
    for f in fingers:
        d = [[np.linalg.norm(rest_pos[f['rod_balls'][i]] - rest_pos[f['servo_data'][j]['horn_ball']])
              for j in range(2)] for i in range(2)]
        if d[0][0] + d[1][1] > d[0][1] + d[1][0]:
            f['servo_data'] = [f['servo_data'][1], f['servo_data'][0]]
            d = [[d[0][1], d[0][0]], [d[1][1], d[1][0]]]
        f['rod_lengths'] = [d[0][0], d[1][1]]
        # distal_dup and distal_real are the same physical piece, exported as two
        # links with different frames: their offset is rigid in the dup frame,
        # not constant in the world, so store it locally and rotate it at solve time.
        # Early versions only matched the ORIGIN point (3 equations) and left
        # orientation unconstrained; the knuckle's single rotation axis (q_knu)
        # can always hit a position target but has no way to also track the
        # orientation implied by the rod-driven side, so proximal/distal visibly
        # twisted apart at rest = ~2, growing to ~50 by 60deg flex. Also
        # matching the rest-pose ROTATION offset (and freeing the rod's twist
        # DOF, previously pinned at 0) lets the solver use all available freedom
        # to minimize that mismatch instead of ignoring it entirely.
        T_dd = hand.fk(f['distal_dup'])
        f['knuckle_offset_local'] = T_dd[:3, :3].T @ (rest_pos[f['distal_real']] - T_dd[:3, 3])
        T_dr = hand.fk(f['distal_real'])
        f['knuckle_rot_offset_local'] = T_dd[:3, :3].T @ T_dr[:3, :3]

        # Rod visuals: each rod end hangs off its ball via a 3-DOF decomposed
        # spherical joint. Record, per chain, the aim point of the lever/rod mesh
        # in the terminal frame (the opposite ball's center at rest), so the
        # chains can be re-aimed after every solve instead of staying at rest
        # and letting the rods swing rigidly with the horns.
        f['rod_chains'] = []
        for i in range(2):
            hb = f['servo_data'][i]['horn_ball']
            rb = f['rod_balls'][i]
            for start, target in ((hb, rb), (rb, hb)):
                joints, terminal = ball_chain(hand, start)
                if len(joints) != 3:
                    continue
                T_t = hand.fk(terminal)
                aim_local = T_t[:3, :3].T @ (rest_pos[target] - T_t[:3, 3])
                f['rod_chains'].append(dict(joints=joints, terminal=terminal,
                                            target_ball=target, aim_local=aim_local))
    return fingers


def _orientation_residual(R_a, R_b):
    """Vee-map of the antisymmetric part of R_a^T R_b: ~0 when the two
    orientations match, grows smoothly with the mismatch angle. Not an exact
    log-map but well-behaved for least_squares over the range this linkage
    actually swings through."""
    R_diff = R_a.T @ R_b
    return np.array([
        R_diff[2, 1] - R_diff[1, 2],
        R_diff[0, 2] - R_diff[2, 0],
        R_diff[1, 0] - R_diff[0, 1],
    ]) * 0.5


def make_residual_fn(hand, f):
    rb0, rb1 = f['rod_balls']
    hb0, hb1 = f['servo_data'][0]['horn_ball'], f['servo_data'][1]['horn_ball']
    act0, act1 = f['servo_data'][0]['actuator_joint'], f['servo_data'][1]['actuator_joint']
    L0, L1 = f['rod_lengths']
    q1j, q2j, propj, knuj, prismj, twistj = (f['q1_joint'], f['q2_joint'], f['prox_joint'],
                                              f['knuckle_joint'], f['prism_joint'], f['twist_joint'])
    distal_real, distal_dup = f['distal_real'], f['distal_dup']
    offset_local = f['knuckle_offset_local']
    rot_offset_local = f['knuckle_rot_offset_local']

    def residual(x, theta_A, theta_B):
        q1, q2, q_prox, q_knu, q_prism, q_twist = x
        p_rb0 = hand.fk(rb0, {q1j: q1, q2j: q2})[:3, 3]
        p_rb1 = hand.fk(rb1, {q1j: q1, q2j: q2})[:3, 3]
        p_hb0 = hand.fk(hb0, {act0: theta_A})[:3, 3]
        p_hb1 = hand.fk(hb1, {act1: theta_B})[:3, 3]
        T_dr = hand.fk(distal_real, {q1j: q1, q2j: q2, prismj: q_prism, twistj: q_twist})
        T_dd = hand.fk(distal_dup, {q1j: q1, propj: q_prox, knuj: q_knu})
        p_target = T_dd[:3, :3] @ offset_local + T_dd[:3, 3]
        R_target = T_dd[:3, :3] @ rot_offset_local
        return np.array([
            np.linalg.norm(p_rb0 - p_hb0) - L0,
            np.linalg.norm(p_rb1 - p_hb1) - L1,
            *(T_dr[:3, 3] - p_target),
            *_orientation_residual(T_dr[:3, :3], R_target),
        ])
    return residual


def make_ik_residual_fn(hand, f):
    """Inverse of make_residual_fn: knuckle command (q1, q_prox) in,
    unknowns are the two servo angles plus the remaining passives."""
    rb0, rb1 = f['rod_balls']
    hb0, hb1 = f['servo_data'][0]['horn_ball'], f['servo_data'][1]['horn_ball']
    act0, act1 = f['servo_data'][0]['actuator_joint'], f['servo_data'][1]['actuator_joint']
    L0, L1 = f['rod_lengths']
    q1j, q2j, propj, knuj, prismj, twistj = (f['q1_joint'], f['q2_joint'], f['prox_joint'],
                                              f['knuckle_joint'], f['prism_joint'], f['twist_joint'])
    distal_real, distal_dup = f['distal_real'], f['distal_dup']
    offset_local = f['knuckle_offset_local']
    rot_offset_local = f['knuckle_rot_offset_local']

    def residual(x, q1, q_prox):
        theta_A, theta_B, q2, q_knu, q_prism, q_twist = x
        p_rb0 = hand.fk(rb0, {q1j: q1, q2j: q2})[:3, 3]
        p_rb1 = hand.fk(rb1, {q1j: q1, q2j: q2})[:3, 3]
        p_hb0 = hand.fk(hb0, {act0: theta_A})[:3, 3]
        p_hb1 = hand.fk(hb1, {act1: theta_B})[:3, 3]
        T_dr = hand.fk(distal_real, {q1j: q1, q2j: q2, prismj: q_prism, twistj: q_twist})
        T_dd = hand.fk(distal_dup, {q1j: q1, propj: q_prox, knuj: q_knu})
        p_target = T_dd[:3, :3] @ offset_local + T_dd[:3, 3]
        R_target = T_dd[:3, :3] @ rot_offset_local
        return np.array([
            np.linalg.norm(p_rb0 - p_hb0) - L0,
            np.linalg.norm(p_rb1 - p_hb1) - L1,
            *(T_dr[:3, 3] - p_target),
            *_orientation_residual(T_dr[:3, :3], R_target),
        ])
    return residual


class HandKinematicsNode(Node):
    def __init__(self):
        super().__init__('hand_kinematics_node')
        self.declare_parameter('robot_description', '')
        self.declare_parameter('command_space', 'knuckle')
        self.declare_parameter('link_prefix', '')
        self.declare_parameter('alias_prefix', '')
        urdf_xml = self.get_parameter('robot_description').value
        if not urdf_xml:
            self.get_logger().error("robot_description parameter is empty; can't build kinematics")
            raise SystemExit(1)
        self.command_space = self.get_parameter('command_space').value
        if self.command_space not in ('servo', 'knuckle'):
            self.get_logger().error(f"command_space must be 'servo' or 'knuckle', got '{self.command_space}'")
            raise SystemExit(1)
        link_prefix = self.get_parameter('link_prefix').value
        alias_prefix = self.get_parameter('alias_prefix').value

        self.hand = Kinematics(urdf_xml)
        fingers = discover_fingers(self.hand, link_prefix=link_prefix)
        self.fingers = calibrate(self.hand, fingers)
        # Deterministic finger numbering, identical for both hands: sort by rest
        # height of the gimbal in the hand's OWN frame (not the combined robot's
        # root) -- fingers 1..3 across the palm top-down, 4 = thumb (its gimbal
        # sits much lower, palm-front). Using the hand's local frame keeps this
        # correct regardless of how the hand is mounted/rotated onto an arm.
        root_link = f'{link_prefix}root'
        T_root_inv = np.linalg.inv(self.hand.fk(root_link))
        self.fingers.sort(key=lambda f: -(T_root_inv @ self.hand.fk(f['gimbal']))[2, 3])
        if self.command_space == 'servo':
            self.residual_fns = [make_residual_fn(self.hand, f) for f in self.fingers]
        else:
            self.residual_fns = [make_ik_residual_fn(self.hand, f) for f in self.fingers]
        self.warm_start = [np.zeros(6) for _ in self.fingers]
        self.rod_warm_start = [[np.zeros(3) for _ in f['rod_chains']] for f in self.fingers]

        # ros2_control-facing alias names: j1<i> = yaw, j2<i> = flexion of finger i
        # (alias_prefix distinguishes left/right when both hands share one
        # combined robot_description, e.g. 'openarm_left_' -> 'openarm_left_j11')
        self.alias_of = {}   # alias -> real joint name
        self.alias_state = {}  # alias -> last solved/clamped value
        for i, f in enumerate(self.fingers, start=1):
            self.alias_of[f'{alias_prefix}j1{i}'] = f['q1_joint']
            self.alias_of[f'{alias_prefix}j2{i}'] = f['prox_joint']
            self.alias_state[f'{alias_prefix}j1{i}'] = 0.0
            self.alias_state[f'{alias_prefix}j2{i}'] = 0.0

        self.get_logger().info(
            f"command_space={self.command_space}, link_prefix='{link_prefix}'; discovered {len(self.fingers)} "
            f"fingers, rod lengths: {[tuple(round(x, 5) for x in f['rod_lengths']) for f in self.fingers]}")
        for i, f in enumerate(self.fingers, start=1):
            self.get_logger().info(
                f"  finger {i}: {alias_prefix}j1{i}=yaw({f['q1_joint']}) {alias_prefix}j2{i}=flex({f['prox_joint']}) "
                f"gimbal={f['gimbal']} servos={f['servo_data'][0]['actuator_joint']},"
                f"{f['servo_data'][1]['actuator_joint']}")

        # Every movable joint of THIS hand must appear in the published joint
        # state -- robot_state_publisher only emits TF for joints it has seen,
        # and the URDF's disconnected passive branches (ball chains,
        # loop-closure stubs) are not part of any finger solve; they stay at
        # the assembled rest pose. Scoped to link_prefix so that, when both
        # hands share one robot_description, each node instance only reports
        # its own side -- otherwise two instances racing on the shared
        # /joint_states topic would each try to zero out the other's joints.
        self.all_movable = [jn for jn, j in self.hand.joints.items()
                            if j['type'] != 'fixed' and jn not in self.alias_of
                            and j['parent'].startswith(link_prefix)]

        joint_commands_topic = self.declare_parameter('joint_commands_topic', 'ahand/joint_commands').value
        joint_states_topic = self.declare_parameter('joint_states_topic', 'ahand/joint_states').value

        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        # queue depth 1: a solve takes tens of ms, so under a fast command stream
        # (e.g. a JTC trajectory sampled at 50 Hz) always solve the newest target
        # and let stale ones drop instead of building a backlog.
        self.sub = self.create_subscription(JointState, 'joint_states_raw', self.on_joint_states, 1)
        # ros2_control bridge (topic_based_ros2_control hardware): commands arrive
        # with alias names; alias states are fed back for the hardware interface.
        self.cmd_sub = self.create_subscription(JointState, joint_commands_topic, self.on_joint_states, 1)
        self.alias_pub = self.create_publisher(JointState, joint_states_topic, 10)
        self.last_full_state = None
        self.on_joint_states(JointState())  # publish the solved rest pose immediately
        # The hardware only publishes commands when they change, so keep both the
        # alias feedback and the full state (TF) alive between commands.
        self.state_timer = self.create_timer(0.05, self.republish_states)

    def republish_states(self):
        now = self.get_clock().now().to_msg()
        msg = JointState()
        msg.header.stamp = now
        msg.name = list(self.alias_state.keys())
        msg.position = list(self.alias_state.values())
        self.alias_pub.publish(msg)
        if self.last_full_state is not None:
            full = JointState()
            full.header.stamp = now
            full.name, full.position = self.last_full_state
            self.pub.publish(full)

    @staticmethod
    def _solve_multistart(res_fn, x0, args, bounds):
        """Least-squares from the warm start; if that lands in a poor local
        minimum (the knuckle linkage has a mirror branch), retry from rest."""
        sol = least_squares(res_fn, x0, args=args, bounds=bounds,
                             xtol=1e-10, ftol=1e-10, max_nfev=200)
        if sol.cost > 1e-9 and np.any(x0 != 0.0):
            sol_rest = least_squares(res_fn, np.zeros_like(x0), args=args, bounds=bounds,
                                      xtol=1e-10, ftol=1e-10, max_nfev=200)
            if sol_rest.cost < sol.cost:
                sol = sol_rest
        return sol

    def solve_finger_servo(self, f, res_fn, x0, raw):
        """Forward: servo angles from raw -> passive joints.
        x = (q1, q2, q_prox, q_knu, q_prism, q_twist)."""
        theta_A = raw.get(f['servo_data'][0]['actuator_joint'], 0.0)
        theta_B = raw.get(f['servo_data'][1]['actuator_joint'], 0.0)
        lower = [f['q1_lower'], -1.2, f['prox_lower'], -1.2, -0.02, -3.2]
        upper = [f['q1_upper'], 1.2, f['prox_upper'], 1.2, 0.02, 3.2]
        sol = self._solve_multistart(res_fn, x0, (theta_A, theta_B), (lower, upper))
        x0[:] = sol.x
        q1, q2, q_prox, q_knu, q_prism, q_twist = sol.x
        return sol, {
            f['q1_joint']: q1, f['q2_joint']: q2, f['prox_joint']: q_prox,
            f['knuckle_joint']: q_knu, f['prism_joint']: q_prism, f['twist_joint']: q_twist,
        }

    def solve_finger_knuckle(self, f, res_fn, x0, raw):
        """Inverse: knuckle command from raw -> servo angles + remaining passives.
        x = (theta_A, theta_B, q2, q_knu, q_prism, q_twist)."""
        q1 = float(np.clip(raw.get(f['q1_joint'], 0.0), f['q1_lower'], f['q1_upper']))
        q_prox = float(np.clip(raw.get(f['prox_joint'], 0.0), f['prox_lower'], f['prox_upper']))
        sd0, sd1 = f['servo_data']
        lower = [sd0['lower'], sd1['lower'], -1.2, -1.2, -0.02, -3.2]
        upper = [sd0['upper'], sd1['upper'], 1.2, 1.2, 0.02, 3.2]
        sol = self._solve_multistart(res_fn, x0, (q1, q_prox), (lower, upper))
        x0[:] = sol.x
        theta_A, theta_B, q2, q_knu, q_prism, q_twist = sol.x
        return sol, {
            sd0['actuator_joint']: theta_A, sd1['actuator_joint']: theta_B,
            f['q1_joint']: q1, f['q2_joint']: q2, f['prox_joint']: q_prox,
            f['knuckle_joint']: q_knu, f['prism_joint']: q_prism, f['twist_joint']: q_twist,
        }

    def solve_rod_chains(self, f, rod_x0s, raw, solved):
        """Aim each rod's ball-joint chain at the opposite ball so the lever/rod
        meshes stay visually connected (they carry no constraint of their own)."""
        vals = dict(raw)
        vals.update(solved)
        ball_pos = {}
        for i in range(2):
            for ball in (f['servo_data'][i]['horn_ball'], f['rod_balls'][i]):
                ball_pos[ball] = self.hand.fk(ball, vals)[:3, 3]
        for chain, x0 in zip(f['rod_chains'], rod_x0s):
            target = ball_pos[chain['target_ball']]
            joints, terminal, aim_local = chain['joints'], chain['terminal'], chain['aim_local']

            def residual(x):
                cv = dict(vals)
                cv.update(zip(joints, x))
                T = self.hand.fk(terminal, cv)
                err = T[:3, :3] @ aim_local + T[:3, 3] - target
                # The point constraint leaves the roll about the rod axis free;
                # the lever mesh is asymmetric, so softly prefer the minimal
                # rotation from rest instead of whatever roll the warm start
                # drifted to (weight small enough not to disturb the point match).
                return np.concatenate([err, 2e-3 * x])

            sol = least_squares(residual, x0, xtol=1e-8, ftol=1e-8, max_nfev=60)
            if np.linalg.norm(sol.x) > 0.3:
                sol_rest = least_squares(residual, np.zeros(3), xtol=1e-8, ftol=1e-8, max_nfev=60)
                if sol_rest.cost <= sol.cost:
                    sol = sol_rest
            x0[:] = sol.x
            solved.update(zip(joints, sol.x))

    def on_joint_states(self, msg: JointState):
        raw = dict(zip(msg.name, msg.position))
        for alias, real in self.alias_of.items():
            if alias in raw:
                raw[real] = raw[alias]
        out_names = [n for n in msg.name if n not in self.alias_of]
        out_pos = [p for n, p in zip(msg.name, msg.position) if n not in self.alias_of]
        by_name = {n: i for i, n in enumerate(out_names)}

        solve = self.solve_finger_servo if self.command_space == 'servo' else self.solve_finger_knuckle
        for f, res_fn, x0, rod_x0s in zip(self.fingers, self.residual_fns, self.warm_start, self.rod_warm_start):
            sol, solved = solve(f, res_fn, x0, raw)
            self.solve_rod_chains(f, rod_x0s, raw, solved)
            for alias, real in self.alias_of.items():
                if real in solved:
                    self.alias_state[alias] = solved[real]
            if sol.cost > 1e-6:
                self.get_logger().warn(
                    f"finger {f['gimbal']}: command out of mechanism reach (residual {sol.cost:.2e}), "
                    f"showing nearest feasible pose", throttle_duration_sec=2.0)
            for jn, v in solved.items():
                if jn in by_name:
                    out_pos[by_name[jn]] = v
                else:
                    by_name[jn] = len(out_names)
                    out_names.append(jn)
                    out_pos.append(v)

        for jn in self.all_movable:
            if jn not in by_name:
                by_name[jn] = len(out_names)
                out_names.append(jn)
                out_pos.append(0.0)

        self.last_full_state = (out_names, out_pos)
        out = JointState()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = out_names
        out.position = out_pos
        self.pub.publish(out)


def main():
    rclpy.init()
    node = HandKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
