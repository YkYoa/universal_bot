"""MediaPipe Pose landmark index constants and the arm+torso connection
subset used for retargeting and for the web overlay.

Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
(33 landmarks; we only need a subset for arm gesture-follow).
"""

NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24

# The landmarks retargeting actually consumes, per side.
ARM_LANDMARKS = {
    "left": {"shoulder": LEFT_SHOULDER, "elbow": LEFT_ELBOW, "wrist": LEFT_WRIST},
    "right": {"shoulder": RIGHT_SHOULDER, "elbow": RIGHT_ELBOW, "wrist": RIGHT_WRIST},
}

TORSO_LANDMARKS = {
    "shoulder_left": LEFT_SHOULDER,
    "shoulder_right": RIGHT_SHOULDER,
    "hip_left": LEFT_HIP,
    "hip_right": RIGHT_HIP,
}

FACING_LANDMARKS = {
    "nose": NOSE,
    "left_ear": LEFT_EAR,
    "right_ear": RIGHT_EAR,
}

# Arm + torso subset of MediaPipe's own POSE_CONNECTIONS (drawing_utils),
# with face/legs dropped - this is what the web overlay draws as bone line
# segments (never a bounding box; see plan Part "skeleton drawing").
ARM_TORSO_CONNECTIONS = (
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
)

# All landmark indices the pipeline needs from a detection; anything else in
# a MediaPipe result is ignored so we do not carry face/leg landmarks around.
REQUIRED_LANDMARK_INDICES = frozenset(
    {NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW,
     RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP}
)
