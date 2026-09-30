import cv2
import time
import csv
import os
import mediapipe as mp

from landmark_extractor import (
    get_elbow_angle,
    get_shoulder_angle,
    get_elbow_horizontal_offset
)


# ==========================================
# 1. Dataset configuration
# ==========================================

DATA_PATH = "data/bicep_curl_dataset.csv"

FIELDNAMES = [
    "left_elbow_angle",
    "left_shoulder_angle",
    "left_elbow_offset",
    "label"
]

# Save approximately 5 samples per second
SAMPLE_INTERVAL = 0.2


# ==========================================
# 2. Create CSV if it doesn't exist
# ==========================================

if not os.path.exists(DATA_PATH):

    with open(DATA_PATH, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()


# ==========================================
# 3. Save one sample
# ==========================================

def save_sample(
    elbow_angle,
    shoulder_angle,
    elbow_offset,
    label
):

    with open(DATA_PATH, "a", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writerow({
            "left_elbow_angle": elbow_angle,
            "left_shoulder_angle": shoulder_angle,
            "left_elbow_offset": elbow_offset,
            "label": label
        })


# ==========================================
# 4. MediaPipe configuration
# ==========================================

MODEL_PATH = "models/pose_landmarker_full.task"

latest_result = None


def result_callback(
    result,
    output_image,
    timestamp_ms
):

    global latest_result

    latest_result = result


BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


# ==========================================
# 5. Pose Landmarker options
# ==========================================

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


# ==========================================
# 6. Start camera
# ==========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("❌ Could not open camera")

    exit()


print("==========================================")
print("     FormFit AI - Data Collector")
print("==========================================")
print("Press 'c' → record CORRECT form")
print("Press 'i' → record INCORRECT form")
print("Press 'q' → quit")
print("==========================================")


# ==========================================
# 7. Recording variables
# ==========================================

active_label = None

last_sample_time = 0

sample_count = 0


# ==========================================
# 8. Start MediaPipe
# ==========================================

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        # ----------------------------------
        # Read camera frame
        # ----------------------------------

        success, frame = cap.read()

        if not success:

            print("❌ Failed to read camera frame")

            break


        # ----------------------------------
        # Flip frame
        # ----------------------------------

        frame = cv2.flip(frame, 1)


        # ----------------------------------
        # Convert BGR → RGB
        # ----------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------
        # Create MediaPipe image
        # ----------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame

        )


        # ----------------------------------
        # Timestamp
        # ----------------------------------

        timestamp_ms = int(
            time.time() * 1000
        )


        # ----------------------------------
        # Send frame to MediaPipe
        # ----------------------------------

        landmarker.detect_async(
            mp_image,
            timestamp_ms
        )


        # ==================================
        # 9. Extract features
        # ==================================

        if (
            latest_result is not None
            and latest_result.pose_landmarks
        ):

            landmarks = latest_result.pose_landmarks[0]


            # Left-side landmarks

            left_shoulder = landmarks[11]

            left_elbow = landmarks[13]

            left_wrist = landmarks[15]

            left_hip = landmarks[23]


            # ----------------------------------
            # Feature 1: Elbow angle
            # ----------------------------------

            left_elbow_angle = get_elbow_angle(
                left_shoulder,
                left_elbow,
                left_wrist
            )


            # ----------------------------------
            # Feature 2: Shoulder angle
            # ----------------------------------

            left_shoulder_angle = get_shoulder_angle(
                left_hip,
                left_shoulder,
                left_elbow
            )


            # ----------------------------------
            # Feature 3: Elbow offset
            # ----------------------------------

            left_elbow_offset = get_elbow_horizontal_offset(
                left_shoulder,
                left_elbow
            )


            # ==================================
            # 10. Automatic data collection
            # ==================================

            current_time = time.time()


            if (
                active_label is not None
                and current_time - last_sample_time
                >= SAMPLE_INTERVAL
            ):

                save_sample(
                    left_elbow_angle,
                    left_shoulder_angle,
                    left_elbow_offset,
                    active_label
                )

                last_sample_time = current_time

                sample_count += 1

                print(
                    f"Saved #{sample_count} | "
                    f"{active_label} | "
                    f"Elbow: {left_elbow_angle:.1f} | "
                    f"Shoulder: {left_shoulder_angle:.1f} | "
                    f"Offset: {left_elbow_offset:.3f}"
                )


            # ==================================
            # 11. Display features
            # ==================================

            cv2.putText(
                frame,
                f"Elbow Angle: {left_elbow_angle:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Shoulder Angle: {left_shoulder_angle:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Elbow Offset: {left_elbow_offset:.3f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ==================================
        # 12. Recording status
        # ==================================

        if active_label == "correct":

            status_text = "RECORDING: CORRECT"

        elif active_label == "incorrect":

            status_text = "RECORDING: INCORRECT"

        else:

            status_text = "NOT RECORDING"


        cv2.putText(
            frame,
            status_text,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        # ==================================
        # 13. Show camera
        # ==================================

        cv2.imshow(
            "FormFit AI - Data Collector",
            frame
        )


        # ==================================
        # 14. Keyboard input
        # ==================================

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------
        # Correct mode
        # ----------------------------------

        if key == ord("c"):

            active_label = "correct"

            print("🟢 Recording CORRECT samples...")


        # ----------------------------------
        # Incorrect mode
        # ----------------------------------

        elif key == ord("i"):

            active_label = "incorrect"

            print("🔴 Recording INCORRECT samples...")


        # ----------------------------------
        # Quit
        # ----------------------------------

        elif key == ord("q"):

            break


# ==========================================
# 15. Release camera
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("==========================================")
print("👋 Data collection stopped.")
print(f"Total samples collected: {sample_count}")
print("==========================================")