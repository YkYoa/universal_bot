# Deploy notes

## 2026-09-16: named joint-state wire format for ExecuteSkill

`ExecuteSkill.action`'s `joint_targets`/`joint_sequence` fields (flat `float64[]`,
implicit fixed stride) were replaced with `sensor_msgs/JointState` name/position
pairs, resolved against the live robot model instead of checked against a
hardcoded/assumed DOF count. Fixes the "joint target has N value(s) but group
needs M" and "flat array with a multiple-of-N length" errors that both_arms
sequences and 7-value waypoints hit under `ee_type:=amazing_hand`.

Touches: `communication/messages/action/ExecuteSkill.action`,
`control/robot_skills/{skill_base.hpp,skill_server.cpp,skills/move_to_joint*}`,
`control/sequence_executor/{skill_client.*,sequence_fsm.cpp}`,
`builds/qvic_2026/src/qvic_actions.cpp`, `builds/openarm_demo/src/openarm_demo_node.cpp`,
`utilities/common/include/common/joint_names.hpp` (new).

### Verifying this landed on a given deployment

```bash
# active_build should be this session's commit or later
curl -s http://127.0.0.1:5050/health | python3 -m json.tool

# joint_names.hpp only exists once this change has synced/built
test -f "$ARM_WS/src/utilities/common/include/common/joint_names.hpp" && echo present || echo MISSING
```
