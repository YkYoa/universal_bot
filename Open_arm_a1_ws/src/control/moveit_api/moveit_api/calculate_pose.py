import numpy as np

# Coordinates: [x, y, z, qx, qy, qz, qw]
p_start = np.array([0.4335, 0.2404, 1.2887])
q = np.array([0.0953, -0.0001, 0.9955, -0.0000]) # [x, y, z, w]

print("=== Global (World) Frame Y-Shift ===")
print(f"Positive Y-shift (+0.1m): [{p_start[0]:.4f}, {p_start[1] + 0.1:.4f}, {p_start[2]:.4f}, {q[0]}, {q[1]}, {q[2]}, {q[3]}]")
print(f"Negative Y-shift (-0.1m): [{p_start[0]:.4f}, {p_start[1] - 0.1:.4f}, {p_start[2]:.4f}, {q[0]}, {q[1]}, {q[2]}, {q[3]}]")

# Local tool frame calculations
# Quaternion rotation matrix
def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2]
    ])

R = quaternion_to_matrix(q)
# Local movement vector in tool frame (y = 0.1m)
v_tool_pos = np.array([0.0, 0.1, 0.0])
v_tool_neg = np.array([0.0, -0.1, 0.0])

# Transform to world frame
v_world_pos = R.dot(v_tool_pos)
v_world_neg = R.dot(v_tool_neg)

p_new_pos = p_start + v_world_pos
p_new_neg = p_start + v_world_neg

print("\n=== Local (Tool) Frame Y-Shift ===")
print(f"Positive Local Y-shift (+0.1m): [{p_new_pos[0]:.4f}, {p_new_pos[1]:.4f}, {p_new_pos[2]:.4f}, {q[0]}, {q[1]}, {q[2]}, {q[3]}]")
print(f"Negative Local Y-shift (-0.1m): [{p_new_neg[0]:.4f}, {p_new_neg[1]:.4f}, {p_new_neg[2]:.4f}, {q[0]}, {q[1]}, {q[2]}, {q[3]}]")
