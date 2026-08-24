"""
SignBridge AI - Improved Data Collection Script
Uses MediaPipe Tasks API for hand landmark detection.

Collects diverse landmark samples for gesture classification.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import csv
import time

from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)
from mediapipe.tasks.python.core.base_options import BaseOptions


# ============================================================
# CONFIG
# ============================================================

CLASSES = [
    "Hello",
    "Yes",
    "No",
    "Help",
    "Thank You",
    "Please",
    "Sorry",
    "Welcome",
    "Stop",
    "Good",
    "Bad",
    "Wait"
]

# 300 samples for every gesture
SAMPLES_PER_CLASS = 300

# Minimum time between two saved samples
SAMPLE_INTERVAL = 0.08

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_PATH = "model/hand_landmarker.task"

NUM_LANDMARKS = 21
FEATURE_COUNT = NUM_LANDMARKS * 3


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9),
    (9, 13),
    (13, 17),
]


os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(hand_landmarks_list):
    """
    Normalize 21 hand landmarks relative to the wrist
    and scale them based on the maximum distance.
    """

    lm = np.array(
        [[l.x, l.y, l.z] for l in hand_landmarks_list],
        dtype=np.float32
    )

    # Wrist = landmark 0
    wrist = lm[0].copy()

    # Move wrist to origin
    lm -= wrist

    # Calculate distances from origin
    distances = np.linalg.norm(lm, axis=1)

    max_dist = distances.max()

    if max_dist > 0:
        lm /= max_dist

    return lm.flatten().tolist()


# ============================================================
# DRAW HAND
# ============================================================

def draw_landmarks(frame, landmarks, w, h):

    pts = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ]

    # Connections
    for a, b in HAND_CONNECTIONS:

        cv2.line(
            frame,
            pts[a],
            pts[b],
            (88, 166, 255),
            2
        )

    # Points
    for i, pt in enumerate(pts):

        if i == 0:
            color = (124, 58, 237)
        else:
            color = (88, 166, 255)

        cv2.circle(
            frame,
            pt,
            5,
            color,
            -1
        )

        cv2.circle(
            frame,
            pt,
            5,
            (255, 255, 255),
            1
        )


# ============================================================
# CHECK MODEL
# ============================================================

def check_model():

    if not os.path.exists(MODEL_PATH):

        print(
            f"\nERROR: Hand landmark model not found at:"
            f"\n{MODEL_PATH}\n"
        )

        print("Run this command:")

        print(
            "python -c "
            "\"import urllib.request; "
            "urllib.request.urlretrieve("
            "'https://storage.googleapis.com/"
            "mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/"
            "hand_landmarker.task', "
            "'model/hand_landmarker.task')\""
        )

        return False

    return True


# ============================================================
# COLLECTION
# ============================================================

def collect_data():

    if not check_model():
        return

    print("\nInitializing MediaPipe...")

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=RunningMode.IMAGE,

        num_hands=1,

        min_hand_detection_confidence=0.6,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)


    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Cannot open webcam.")

        landmarker.close()

        return


    # Try to use HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    # ========================================================
    # CSV
    # ========================================================

    fieldnames = [
        "label"
    ] + [
        f"f{i}"
        for i in range(FEATURE_COUNT)
    ]

    file_exists = os.path.exists(CSV_PATH)


    existing_counts = {
        cls: 0
        for cls in CLASSES
    }


    # Read existing dataset if present
    if file_exists:

        try:

            df = pd.read_csv(CSV_PATH)

            if "label" in df.columns:

                for cls in CLASSES:

                    existing_counts[cls] = int(
                        (df["label"] == cls).sum()
                    )

        except Exception as e:

            print(
                f"Warning: Could not read existing dataset: {e}"
            )


    csv_file = open(
        CSV_PATH,
        "a",
        newline=""
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    # Write header only if file is empty
    if (
        not file_exists
        or os.path.getsize(CSV_PATH) == 0
    ):

        writer.writeheader()


    # ========================================================
    # COLLECTION LOOP
    # ========================================================

    for cls in CLASSES:

        existing = existing_counts[cls]

        needed = max(
            0,
            SAMPLES_PER_CLASS - existing
        )


        if needed == 0:

            print(
                f"[SKIP] {cls}: "
                f"already has {existing} samples"
            )

            continue


        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        print("\n" + "=" * 60)

        print(
            f"COLLECTING: {cls}"
        )

        print(
            f"Target: {SAMPLES_PER_CLASS} samples"
        )

        print(
            f"Already available: {existing}"
        )

        print(
            f"Need: {needed} more samples"
        )

        print("=" * 60)

        print(
            "\nIMPORTANT:"
        )

        print(
            "Keep performing the SAME gesture."
        )

        print(
            "Slowly move your hand:"
        )

        print(
            "  - slightly left/right"
        )

        print(
            "  - slightly up/down"
        )

        print(
            "  - slightly closer/farther"
        )

        print(
            "  - slightly rotate your wrist"
        )

        print(
            "\nDo NOT change the actual gesture."
        )

        print(
            "\nPress Q anytime to stop."
        )


        # ----------------------------------------------------
        # Countdown
        # ----------------------------------------------------

        countdown_start = time.time()

        while (
            time.time() - countdown_start
            < 4.0
        ):

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            remaining = 4 - int(
                time.time() - countdown_start
            )

            cv2.putText(
                frame,
                f"NEXT: {cls}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                f"Starting in {remaining}...",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Move your hand naturally while maintaining the gesture",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

            cv2.imshow(
                "SignBridge AI - Data Collection",
                frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                cap.release()
                cv2.destroyAllWindows()
                csv_file.close()
                landmarker.close()

                return


        # ----------------------------------------------------
        # Actual collection
        # ----------------------------------------------------

        collected = 0

        last_saved_time = 0


        while collected < needed:

            ret, frame = cap.read()

            if not ret:

                print(
                    "Frame read error."
                )

                break


            frame = cv2.flip(
                frame,
                1
            )

            h, w = frame.shape[:2]


            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(
                mp_img
            )


            status_color = (
                0,
                0,
                255
            )

            status_text = (
                "NO HAND DETECTED"
            )


            # ------------------------------------------------
            # Hand detected
            # ------------------------------------------------

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                draw_landmarks(
                    frame,
                    hand,
                    w,
                    h
                )


                features = normalize_landmarks(
                    hand
                )


                # ------------------------------------------------
                # Time-based sampling
                # ------------------------------------------------

                current_time = time.time()

                if (
                    len(features)
                    == FEATURE_COUNT
                    and
                    current_time - last_saved_time
                    >= SAMPLE_INTERVAL
                ):

                    row = {
                        "label": cls
                    }

                    row.update({
                        f"f{i}": round(
                            features[i],
                            6
                        )
                        for i in range(
                            FEATURE_COUNT
                        )
                    })


                    writer.writerow(row)

                    csv_file.flush()


                    collected += 1

                    last_saved_time = current_time


                    status_color = (
                        0,
                        255,
                        0
                    )

                    status_text = (
                        f"CAPTURED: "
                        f"{collected}/{needed}"
                    )

                else:

                    status_color = (
                        0,
                        200,
                        255
                    )

                    status_text = (
                        f"Collecting... "
                        f"{collected}/{needed}"
                    )


            # ------------------------------------------------
            # HUD
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"CLASS: {cls}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                status_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )


            # Progress bar
            progress = int(
                (collected / needed) * 300
            )


            cv2.rectangle(
                frame,
                (20, 110),
                (320, 130),
                (50, 50, 50),
                -1
            )


            cv2.rectangle(
                frame,
                (20, 110),
                (20 + progress, 130),
                (0, 200, 0),
                -1
            )


            cv2.putText(
                frame,
                f"{collected}/{needed}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1
            )


            cv2.putText(
                frame,
                "Move hand slowly while maintaining gesture",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )


            cv2.putText(
                frame,
                "Press Q to quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )


            cv2.imshow(
                "SignBridge AI - Data Collection",
                frame
            )


            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                cap.release()
                cv2.destroyAllWindows()
                csv_file.close()
                landmarker.close()

                print(
                    "\nCollection stopped by user."
                )

                return


        print(
            f"\nDONE: {cls} -> "
            f"{existing + collected} total samples"
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    csv_file.close()

    landmarker.close()


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "COLLECTION COMPLETE"
    )

    print("=" * 60)


    df = pd.read_csv(
        CSV_PATH
    )


    for cls in CLASSES:

        count = len(
            df[
                df["label"] == cls
            ]
        )

        print(
            f"  {cls:<12} : {count} samples"
        )


    print(
        f"\nTOTAL: {len(df)} samples"
    )

    print(
        f"SAVED TO: {CSV_PATH}"
    )

    print("=" * 60)

    print(
        "\nNext step:"
    )

    print(
        "python train_model.py"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "SignBridge AI - Improved Data Collection"
    )

    print("=" * 60)

    print(
        f"\nGestures: {len(CLASSES)}"
    )

    print(
        f"Samples per gesture: {SAMPLES_PER_CLASS}"
    )

    print(
        f"Total target samples: "
        f"{len(CLASSES) * SAMPLES_PER_CLASS}"
    )

    print(
        "\nSampling interval:",
        SAMPLE_INTERVAL,
        "seconds"
    )

    print("=" * 60)

    collect_data()