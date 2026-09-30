import cv2
import time
# pyrefly: ignore [missing-import]
import mediapipe as mp

from landmark_extractor import (
    get_elbow_angle,
    get_shoulder_angle,
    get_elbow_horizontal_offset
)

from bicep_curl_counter import BicepCurlCounter

MODEL_PATH = "models/pose_landmarker_full.task"
latest_result = None
counter = BicepCurlCounter()

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("✅ Camera started!")
print("✅ Pose Landmarker started!")
print("Press 'q' to quit.")

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            print("❌ Failed to read camera frame")
            break

        # Mirror camera like a normal selfie camera
        frame = cv2.flip(frame, 1)

        # OpenCV uses BGR
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp must continuously increase
        timestamp_ms = int(time.time() * 1000)

        # Send frame to Pose Landmarker
        landmarker.detect_async(
            mp_image,
            timestamp_ms
        )

        # -----------------------------
        # Draw detected landmarks
        # -----------------------------
        if latest_result is not None:

            if latest_result.pose_landmarks:

                landmarks = latest_result.pose_landmarks[0]
                left_shoulder = landmarks[11]
                left_elbow = landmarks[13]
                left_wrist = landmarks[15]
                left_elbow_offset = get_elbow_horizontal_offset(
                    left_shoulder,
                    left_elbow
                )   

                left_shoulder_angle = get_shoulder_angle(
                    landmarks[23],
                    left_shoulder,
                    left_elbow
                )   

                left_elbow_angle = get_elbow_angle(
                    left_shoulder,
                    left_elbow,
                    left_wrist
                )       

                right_shoulder = landmarks[12]
                right_elbow = landmarks[14]
                right_wrist = landmarks[16]

                right_elbow_angle = get_elbow_angle(
                    right_shoulder,
                    right_elbow,
                    right_wrist
                )

                left_hip = landmarks[23]

                left_elbow_angle = get_elbow_angle(
                    left_shoulder,
                    left_elbow,
                    left_wrist
                )

                left_shoulder_angle = get_shoulder_angle(
                    left_hip,
                    left_shoulder,
                    left_elbow
                )           

                right_shoulder = landmarks[12]
                right_elbow = landmarks[14]
                right_wrist = landmarks[16]

                right_elbow_angle = get_elbow_angle(
                    right_shoulder,
                    right_elbow,
                    right_wrist
                ) 
                reps, state, smoothed_angle = counter.update(left_elbow_angle)
                # Draw pose landmarks
                height, width, _ = frame.shape

                for landmark in landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    # Draw landmark point
                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        frame,
                        f"Left Elbow: {left_elbow_angle:.1f} degrees",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                            2
                    )
                    cv2.putText(
                        frame,
                        f"Elbow Offset: {left_elbow_offset:.3f}",
                        (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 165, 0),
                        2
                    )   

                    cv2.putText(
                        frame,
                        f"Right Elbow: {right_elbow_angle:.1f} degrees",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )  
                    cv2.putText(
                        frame,
                        f"Reps: {reps}",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )  
                    cv2.putText(
                        frame,
                        f"Smoothed Angle: {smoothed_angle:.1f}",
                        (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"State: {state}",
                        (30, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    ) 
                    cv2.putText(
                        frame,
                        f"Left Shoulder: {left_shoulder_angle:.1f} degrees",
                        (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                        (0, 255, 255),
                        2
                    )

        # Display
        cv2.imshow(
            "FormFit AI - Pose Detection",
            frame
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("Pose detection stopped.")