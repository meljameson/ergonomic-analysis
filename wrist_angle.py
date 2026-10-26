#!/usr/bin/env python3
"""Measure 2D wrist angle (forearm vs. hand) from an OAK-D Lite or a video file.

Point the camera perpendicular to the plane of wrist bending:
  - side view  -> flexion / extension
  - top view   -> radial / ulnar deviation

Keys:
  n  capture neutral posture (hold the wrist neutral for ~1 s)
  r  start / stop recording a trial to the CSV
  q  quit
"""
import argparse
import csv
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
HAND_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# MediaPipe Pose landmark indices: (elbow, wrist)
LANDMARKS = {"left": (13, 15), "right": (14, 16)}
# MediaPipe Hand landmark indices: wrist, middle-finger knuckle (axis of the 3rd metacarpal)
HAND_WRIST, HAND_MIDDLE_MCP = 0, 9
MIN_VISIBILITY = 0.5
NEUTRAL_FRAMES = 30
# Hand crop half-size as a fraction of forearm length; big enough to hold a fully bent hand
CROP_SCALE = 1.4
# Run the (slow) pose model on every Nth frame; the hand model runs on every frame
POSE_EVERY = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=["left", "right"], default="right", help="arm to track (the subject's side)")
    p.add_argument("--video", help="analyze a video file instead of the live camera")
    p.add_argument("--out", help="CSV output path (default: trial_<timestamp>.csv)")
    p.add_argument("--save-video", action="store_true", help="also save the annotated video next to the CSV")
    p.add_argument("--flip-sign", action="store_true", help="invert the angle sign so extension is positive")
    p.add_argument("--threshold", type=float, default=15.0, help="highlight angles beyond this many degrees")
    return p.parse_args()


def ensure_models():
    for url, path in ((MODEL_URL, MODEL_PATH), (HAND_MODEL_URL, HAND_MODEL_PATH)):
        if not path.exists():
            print(f"Downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)


def frames_from_camera():
    """Yield (bgr_frame, timestamp_ms) from the OAK-D Lite colour camera."""
    import depthai as dai

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        # The pose model works at 256x256 and the hand model at 224x224, so 640x360 is plenty
        # and keeps USB bandwidth and per-frame copies low (1280x720 dropped the frame rate a lot).
        # Keep only the newest frame so a slow host never falls behind the camera.
        output = cam.requestOutput((640, 360), dai.ImgFrame.Type.BGR888i, fps=30)
        queue = output.createOutputQueue(maxSize=1, blocking=False)
        pipeline.start()
        print("USB speed:", pipeline.getDefaultDevice().getUsbSpeed().name)
        while pipeline.isRunning():
            msg = queue.get()
            yield msg.getCvFrame(), int(msg.getTimestamp().total_seconds() * 1000)


def frames_from_video(path):
    """Yield (bgr_frame, timestamp_ms) from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, int(index * 1000 / fps)
        index += 1
    cap.release()


def arm_points(pose_result, side, width, height):
    """(elbow, wrist) in pixels from the pose model, or None if they aren't visible enough."""
    if not pose_result.pose_landmarks:
        return None
    lms = [pose_result.pose_landmarks[0][i] for i in LANDMARKS[side]]
    if min(lm.visibility for lm in lms) < MIN_VISIBILITY:
        return None
    return [(lm.x * width, lm.y * height) for lm in lms]


def hand_crop_box(elbow, wrist, width, height):
    """Square (x0, y0, x1, y1) around the wrist for the hand model, or None if it's off-frame.

    The body pose model's hand points are too coarse to follow wrist bending, and at camera
    distance the hand is too small for the hand model to find in the full frame, so crop first.
    """
    half = max(int(CROP_SCALE * math.dist(elbow, wrist)), 64)
    x0, y0 = max(int(wrist[0]) - half, 0), max(int(wrist[1]) - half, 0)
    x1, y1 = min(int(wrist[0]) + half, width), min(int(wrist[1]) + half, height)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def detect_hands(hand_landmarker, rgb, box, ts_ms):
    """Run the hand model on a crop; returns each hand's landmarks in full-frame pixels."""
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    result = hand_landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), ts_ms)
    cw, ch = x1 - x0, y1 - y0
    return [[(x0 + lm.x * cw, y0 + lm.y * ch) for lm in hand] for hand in result.hand_landmarks]


def wrist_angle(arm, hands):
    """Signed angle (deg) between forearm (elbow->wrist) and hand (wrist->middle knuckle).

    Returns (angle, points) or (None, None) if the arm or hand can't be found.
    Coordinates are in pixels so the image aspect ratio doesn't skew the angle.
    """
    if arm is None:
        return None, None
    elbow, pose_wrist = arm
    forearm = math.dist(elbow, pose_wrist)
    # Both hands can be in the crop (e.g. carrying a panel); take the one whose wrist matches the pose
    # The pose wrist is loose (it often sits near the knuckles), so allow most of a forearm length
    hands = [h for h in hands if math.dist(h[HAND_WRIST], pose_wrist) < 0.8 * forearm]
    if not hands:
        return None, None
    hand = min(hands, key=lambda h: math.dist(h[HAND_WRIST], pose_wrist))
    wrist, knuckles = hand[HAND_WRIST], hand[HAND_MIDDLE_MCP]

    u = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    v = (knuckles[0] - wrist[0], knuckles[1] - wrist[1])
    angle = math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]))
    return angle, (elbow, wrist, knuckles)


def draw_overlay(frame, points, relative, status, threshold, fps):
    if points:
        elbow, wrist, knuckles = [tuple(int(c) for c in p) for p in points]
        cv2.line(frame, elbow, wrist, (255, 200, 0), 3)
        cv2.line(frame, wrist, knuckles, (0, 200, 255), 3)
        for p in (elbow, wrist, knuckles):
            cv2.circle(frame, p, 6, (255, 255, 255), -1)

    if relative is None:
        text, colour = "angle: --", (200, 200, 200)
    else:
        text = f"angle: {relative:+.1f} deg"
        colour = (0, 0, 255) if abs(relative) > threshold else (0, 220, 0)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)
    cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{fps:.0f} fps", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    args = parse_args()
    ensure_models()

    out_path = Path(args.out or f"trial_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    sign = -1.0 if args.flip_sign else 1.0

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    # VIDEO mode tracks the hand between frames and skips palm detection, roughly halving its cost.
    # The crop follows the wrist, so the hand stays put in crop coordinates and tracking holds.
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        # Must be 1: palm detection keeps re-running every frame while fewer than num_hands are tracked
        num_hands=1,
    )
    # The hand model runs on this thread while the pose model runs on the main one
    hand_executor = ThreadPoolExecutor(max_workers=1)
    crop_box = None
    arm = None
    frame_index = 0

    neutral = None
    calibrating = False
    neutral_samples = []
    recording = False
    trial = 0
    video_writer = None
    last_ts = -1
    fps = 0.0
    last_frame_time = time.monotonic()

    frames = frames_from_video(args.video) if args.video else frames_from_camera()

    with vision.PoseLandmarker.create_from_options(options) as landmarker, \
            vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
            open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "time_s", "raw_angle_deg", "angle_deg", "tracked"])
        trial_start_ms = 0

        for frame, ts_ms in frames:
            # MediaPipe requires strictly increasing timestamps
            ts_ms = max(ts_ms, last_ts + 1)
            last_ts = ts_ms

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Crop for the hand comes from the previous frame's wrist so both models can run at once;
            # the wrist moves only a little in one frame and the crop has plenty of margin.
            hand_future = (hand_executor.submit(detect_hands, hand_landmarker, rgb, crop_box, ts_ms)
                           if crop_box else None)
            # Pose is the slow model and only supplies the elbow, which barely moves while the
            # hand works, so run it every few frames (or every frame until the arm is found)
            if arm is None or frame_index % POSE_EVERY == 0:
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
                arm = arm_points(result, args.side, width, height)
                # Keep the last crop when the pose drops the arm for a frame, so hand tracking isn't reset
                if arm:
                    crop_box = hand_crop_box(*arm, width, height) or crop_box
            frame_index += 1
            hands = hand_future.result() if hand_future else []

            raw, points = wrist_angle(arm, hands)
            if raw is not None:
                raw *= sign

            if calibrating and raw is not None:
                neutral_samples.append(raw)
                if len(neutral_samples) >= NEUTRAL_FRAMES:
                    neutral = sum(neutral_samples) / len(neutral_samples)
                    calibrating = False
                    print(f"Neutral set: {neutral:+.1f} deg (raw)")

            relative = raw - neutral if (raw is not None and neutral is not None) else None

            if recording:
                writer.writerow([
                    trial,
                    f"{(ts_ms - trial_start_ms) / 1000:.3f}",
                    "" if raw is None else f"{raw:.2f}",
                    "" if relative is None else f"{relative:.2f}",
                    int(raw is not None),
                ])

            if calibrating:
                status = "calibrating neutral... hold still"
            elif neutral is None:
                status = "press n to set neutral"
            elif recording:
                status = f"RECORDING trial {trial}  (r to stop)"
            else:
                status = "r to record, n to recalibrate, q to quit"
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_frame_time, 1e-6)
            last_frame_time = now
            draw_overlay(frame, points, relative, status, args.threshold, fps)

            if args.save_video:
                if video_writer is None:
                    video_writer = cv2.VideoWriter(str(out_path.with_suffix(".mp4")),
                                                   cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
                video_writer.write(frame)

            cv2.imshow("wrist angle", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                calibrating = True
                neutral_samples = []
                print("Hold the wrist neutral...")
            if key == ord("r"):
                if neutral is None:
                    print("Set neutral (n) before recording.")
                else:
                    recording = not recording
                    if recording:
                        trial += 1
                        trial_start_ms = ts_ms
                        print(f"Recording trial {trial}")
                    else:
                        print(f"Stopped trial {trial}")

    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
