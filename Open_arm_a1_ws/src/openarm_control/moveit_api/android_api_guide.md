# 📱 OpenArm REST API Integration Guide (Android Team)

This guide documents how the Android UI application can trigger and control the bimanual robot's predefined motion sequences (**Wave**, **Greet**, and **Home Both**) via the REST API server.

---

## 🌐 Base Configuration

* **Protocol**: `HTTP`
* **Default Port**: `5050`
* **Base URL**: `http://<robot-ip>:5050`
* **Headers**: `Content-Type: application/json`

---

## 🎯 Primary Endpoints

All predefined motion sequences are run using a single, unified POST request.

### 1. Run Predefined Sequence
* **Endpoint**: `POST /api/sequence`
* **Content-Type**: `application/json`

#### 📦 Request Payloads

##### 👋 Option A: Trigger "Wave" Sequence
Triggers a custom joint-based wave motion using the left arm (`laWaves_1` to `laWaves_3`).
```json
{
  "name": "wave",
  "loop_count": 1,
  "velocity_scaling": 1.0
}
```

##### 🤝 Option B: Trigger "Greet" Sequence
Triggers a greeting gesture.
```json
{
  "name": "greet",
  "loop_count": 1,
  "velocity_scaling": 1.0
}
```

##### 🏠 Option C: Trigger "Home Both" Sequence
Moves both the left and right arms simultaneously back to their safe home positions.
```json
{
  "name": "home_both",
  "loop_count": 1,
  "velocity_scaling": 0.5
}
```

#### 📥 Query Parameters (JSON Fields)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | String | **Yes** | The name of the sequence (`"wave"`, `"greet"`, or `"home_both"`) |
| `loop_count` | Integer | No | Number of times to repeat (Default: `1`). Set to `0` or `-1` for infinite loop. |
| `velocity_scaling` | Double | No | Overall speed multiplier override (`0.1` to `1.0`). |

---

### 🛑 2. Emergency Stop
Instantly cancels any currently running sequence execution loops and stops the arms.
* **Endpoint**: `POST /api/stop`
* **Request Body**: (None/Empty)

---

## 🕹 Direct / Manual Control

Besides the 3 predefined sequences above, you can now move **any planning group directly** — full arm, one hand, or the head — either the whole group at once or one joint at a time. Useful for a settings/debug screen, per-joint calibration, or a slider-based control UI. All motion still goes through MoveIt (collision-checked): if it would hit something or can't be planned, you get `success: false` back, nothing moves.

### Planning Groups

| Group | Joint count | What it is |
| :--- | :--- | :--- |
| `left_arm` | 7 | left arm only |
| `right_arm` | 7 | right arm only |
| `both_arms` | 14 | both arms, synced (left 7 then right 7) |
| `left_hand_fingers` | 8 | left amazing_hand fingers |
| `right_hand_fingers` | 8 | right amazing_hand fingers |
| `head` | 2 | neck (pan), then head (tilt) |

Full joint name + order for any group: `GET /api/docs` → `planning_groups` (always live, matches whatever's actually deployed).

### Units
All joint values default to **radians**. Add `"unit": "deg"` to use degrees instead — much easier to build a UI around.

### 3. Move One Joint (recommended for a per-joint slider)
Moves exactly one joint; every other joint in that group stays exactly where it was.
* **Endpoint**: `POST /api/move/joint`
```json
{
  "group": "left_arm",
  "joint": "openarm_left_joint4",
  "value": 45,
  "unit": "deg"
}
```
`joint` can be the joint name (see table above via `/api/docs`) **or** a 0-based index (`3` = the 4th joint in that group's order). Example — tilt the head down 20° without touching pan:
```json
{"group": "head", "joint": 1, "value": -20, "unit": "deg"}
```

### 4. Move a Whole Group at Once
* **Endpoint**: `POST /api/move/joints`
```json
{
  "group": "left_arm",
  "positions": [0, -10, 0, 45, 0, 30, 0],
  "unit": "deg",
  "velocity_scaling": 0.3
}
```
`positions` needs exactly one value per joint in the group, in order.

### 5. Named Poses (arms + hands)
* **Endpoint**: `POST /api/move/named`
```json
{"group": "left_hand_fingers", "pose": "open"}
```
Valid `pose` per group: arms → `home` / `ready`. Hands (`left_hand_fingers`/`right_hand_fingers`) → `open` / `close` / `home`. Head has no named poses yet — use `/api/move/joint` or `/api/move/joints`.

### 6. Cartesian Pose (arms only)
* **Endpoint**: `POST /api/move/pose`
```json
{
  "group": "left_arm",
  "position": {"x": 0.4, "y": 0.2, "z": 1.1},
  "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}
}
```

### 7. amazing_hand Shortcut
* **Endpoint**: `POST /api/hand`
```json
{"side": "left", "action": "close"}
```
or with raw values:
```json
{"side": "left", "positions": [0,0,0,0, 1,1,1,1]}
```

---

## 📄 Response Formats

### ✅ Success Response (200 OK / 201 Created)
```json
{
  "success": true,
  "message": "Sequence \"wave\" completed successfully for 1 loop(s).",
  "step_results": [
    {
      "iteration": 1,
      "step": 0,
      "name": "laWaves_1",
      "success": true
    },
    {
      "iteration": 1,
      "step": 1,
      "name": "laWaves_2",
      "success": true
    },
    {
      "iteration": 1,
      "step": 2,
      "name": "laWaves_3",
      "success": true
    }
  ]
}
```

### ❌ Error Response (422 Unprocessable Entity / 500 Internal Error)
Returned if planning fails, joint limits are violated, or a hardware timeout occurs.
```json
{
  "success": false,
  "message": "Sequence failed at iteration 1, step 2",
  "step_results": [
    {
      "iteration": 1,
      "step": 0,
      "name": "laWaves_1",
      "success": true
    },
    {
      "iteration": 1,
      "step": 1,
      "name": "laWaves_2",
      "success": false
    }
  ]
}
```

---

## 💻 Kotlin Integration Example (Retrofit)

Below is the recommended clean implementation for your Android application using Retrofit:

### 1. Define Request Models
```kotlin
data class SequenceRequest(
    val name: String,
    val loop_count: Int = 1,
    val velocity_scaling: Double = 1.0
)

data class StepResult(
    val iteration: Int,
    val step: Int,
    val name: String,
    val success: Boolean
)

data class SequenceResponse(
    val success: Boolean,
    val message: String,
    val step_results: List<StepResult>?
)
```

### 1b. Request Models for Direct/Manual Control
```kotlin
data class MoveJointRequest(
    val group: String,       // "left_arm" / "right_arm" / "both_arms" /
                              // "left_hand_fingers" / "right_hand_fingers" / "head"
    val joint: String,       // joint name, or index as a string e.g. "3"
    val value: Double,
    val unit: String = "deg" // "deg" or "rad"
)

data class MoveJointsRequest(
    val group: String,
    val positions: List<Double>,
    val unit: String = "deg",
    val velocity_scaling: Double = 0.3
)

data class NamedPoseRequest(
    val group: String,
    val pose: String          // "home"/"ready" for arms, "open"/"close"/"home" for hands
)

data class MoveResponse(
    val success: Boolean,
    val message: String
)
```

### 2. Define API Interface
```kotlin
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface RobotApiService {
    @POST("/api/sequence")
    suspend fun runSequence(
        @Body request: SequenceRequest
    ): Response<SequenceResponse>

    @POST("/api/stop")
    suspend fun emergencyStop(): Response<SequenceResponse>

    @POST("/api/move/joint")
    suspend fun moveJoint(
        @Body request: MoveJointRequest
    ): Response<MoveResponse>

    @POST("/api/move/joints")
    suspend fun moveJoints(
        @Body request: MoveJointsRequest
    ): Response<MoveResponse>

    @POST("/api/move/named")
    suspend fun moveNamed(
        @Body request: NamedPoseRequest
    ): Response<MoveResponse>
}
```

### 3. Usage Example inside ViewModel
```kotlin
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

class RobotControlViewModel(private val apiService: RobotApiService) : ViewModel() {

    fun triggerWaveSequence() {
        viewModelScope.launch {
            try {
                val response = apiService.runSequence(SequenceRequest(name = "wave"))
                if (response.isSuccessful && response.body()?.success == true) {
                    // Update UI: Wave sequence succeeded
                } else {
                    // Update UI: Show failure message
                }
            } catch (e: Exception) {
                // Handle network or connection exception
            }
        }
    }

    // Example: per-joint slider, e.g. head tilt slider from -30 to +30 degrees
    fun setHeadTilt(degrees: Double) {
        viewModelScope.launch {
            try {
                val response = apiService.moveJoint(
                    MoveJointRequest(group = "head", joint = "1", value = degrees, unit = "deg")
                )
                if (response.isSuccessful && response.body()?.success == true) {
                    // Update UI: head moved
                } else {
                    // Update UI: show response.body()?.message (e.g. collision, out of range)
                }
            } catch (e: Exception) {
                // Handle network or connection exception
            }
        }
    }
}
```
