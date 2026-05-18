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
}
```
