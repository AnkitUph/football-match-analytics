"""
Roboflow keypoint-detection call, shared by:
  - apps.matches.tasks._run_automatic_calibration (the real, no-human
    path that runs for every match)
  - apps.matches.views.calibrate_suggest (the hidden /calibrate/ page's
    manual smart-assist button, for debugging a bad auto-calibration)

Kept as ONE function so the class_id+1 mapping fix (see
pitch_landmarks_32.py's docstring — the "class" string field is a
Roboflow dataset display label that drifted out of sync with the real
class_id ordering, confirmed via live inference + homography
reprojection testing against two independent clips) only exists in one
place. A second copy would risk silently drifting out of sync if either
path were fixed without the other.
"""
import base64

import cv2
import requests

from apps.matches.pitch_landmarks_32 import ROBOFLOW_KEYPOINTS_32

ROBOFLOW_MODEL_ID = "football-field-detection-f07vi/14"


def detect_pitch_keypoints(frame_bgr, api_key, confidence_threshold=0.5, max_points=12):
    """
    Runs the hosted Roboflow keypoint model on one BGR frame (as read by
    cv2.VideoCapture) and returns up to max_points suggested pixel<->pitch
    correspondences, highest-confidence first.

    Returns a list of dicts: {landmark_id, label, pixel_x, pixel_y,
    pitch_x, pitch_y, confidence}. Empty list on any failure (network
    error, no pitch detected, nothing above threshold) — callers treat
    that as "no suggestion available" and must never raise on it, since
    both call sites (automatic calibration and the manual smart-assist
    button) need to degrade gracefully rather than fail the caller's
    whole operation over a Roboflow/network hiccup.

    IMPORTANT: roboflow_index = class_id + 1. Confirmed via live
    inference AND homography reprojection testing (halfway line, center
    circle, and touchlines landed correctly on real pitch markings
    across two independent clips — test_11.mp4 and test_1.mp4/Mainz)
    that this model's array position (class_id, 0-indexed) is the
    correct index into ROBOFLOW_KEYPOINTS_32 — matching the documented
    usage pattern in sports.common.view.ViewTransformer examples. The
    "class" string field (e.g. "14") must NOT be used — it silently
    produces a badly wrong homography (a tiny, misplaced center circle;
    a halfway line running diagonally across the frame).
    """
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return []

    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    try:
        resp = requests.post(
            f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}",
            params={"api_key": api_key},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        result = resp.json()
    except Exception:
        return []

    predictions = result.get("predictions")
    if not predictions:
        return []

    landmarks_by_index = {l["roboflow_index"]: l for l in ROBOFLOW_KEYPOINTS_32}

    h_frame, w_frame = frame_bgr.shape[:2]
    suggestions = []
    for kp in predictions[0].get("keypoints", []):
        confidence = kp.get("confidence", 0)
        if confidence < confidence_threshold:
            continue
        # Ensure point is strictly inside visible frame boundaries
        kx, ky = kp.get("x", -1), kp.get("y", -1)
        if kx < 0 or kx >= w_frame or ky < 0 or ky >= h_frame:
            continue
        roboflow_index = kp["class_id"] + 1  # NOT kp["class"] — see docstring above
        landmark = landmarks_by_index.get(roboflow_index)
        if landmark is None:
            continue
        suggestions.append({
            "landmark_id": landmark["id"],
            "label": landmark["label"],
            "pixel_x": kx,
            "pixel_y": ky,
            "pitch_x": landmark["x"],
            "pitch_y": landmark["y"],
            "confidence": confidence,
        })

    suggestions.sort(key=lambda s: -s["confidence"])
    return suggestions[:max_points]