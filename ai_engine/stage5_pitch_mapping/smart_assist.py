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
import logging
from pathlib import Path

import cv2
import requests

from apps.matches.pitch_landmarks_32 import ROBOFLOW_KEYPOINTS_32

_logger = logging.getLogger(__name__)
ROBOFLOW_MODEL_ID = "football-field-detection-f07vi/14"
_LOCAL_PITCH_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pitch_keypoints_best.pt"
_LOCAL_PITCH_MODEL = None


def _get_local_pitch_model():
    global _LOCAL_PITCH_MODEL
    if _LOCAL_PITCH_MODEL is None and _LOCAL_PITCH_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            _LOCAL_PITCH_MODEL = YOLO(str(_LOCAL_PITCH_MODEL_PATH))
            _logger.info("Loaded local pitch keypoint model: %s", _LOCAL_PITCH_MODEL_PATH)
        except Exception as e:
            _logger.warning("Could not load local pitch keypoint model: %s", e)
    return _LOCAL_PITCH_MODEL


def detect_pitch_keypoints(frame_bgr, api_key=None, confidence_threshold=0.35, max_points=12):
    """
    Detects up to max_points suggested pixel<->pitch correspondences,
    highest-confidence first.

    Tries local offline pitch keypoint model (pitch_keypoints_best.pt) first.
    Falls back to hosted Roboflow keypoint model if local model is unavailable
    or returns fewer than 4 keypoints.
    """
    landmarks_by_index = {l["roboflow_index"]: l for l in ROBOFLOW_KEYPOINTS_32}
    h_frame, w_frame = frame_bgr.shape[:2]

    # --- Mode 1: Local YOLOv8-Pose Pitch Keypoint Model (Zero API latency) ---
    local_model = _get_local_pitch_model()
    if local_model is not None:
        try:
            results = local_model.predict(
                frame_bgr,
                conf=min(0.20, confidence_threshold),
                verbose=False
            )[0]
            if results.keypoints is not None and len(results.keypoints.data) > 0:
                kpts = results.keypoints.data[0].cpu().numpy()
                suggestions = []
                for idx, (kx, ky, conf) in enumerate(kpts):
                    if conf < confidence_threshold:
                        continue
                    if kx < 0 or kx >= w_frame or ky < 0 or ky >= h_frame:
                        continue
                    roboflow_index = idx + 1
                    landmark = landmarks_by_index.get(roboflow_index)
                    if landmark is None:
                        continue
                    suggestions.append({
                        "landmark_id": landmark["id"],
                        "label": landmark["label"],
                        "pixel_x": float(kx),
                        "pixel_y": float(ky),
                        "pitch_x": landmark["x"],
                        "pitch_y": landmark["y"],
                        "confidence": float(conf),
                    })
                if len(suggestions) >= 4:
                    suggestions.sort(key=lambda s: -s["confidence"])
                    return suggestions[:max_points]
        except Exception as e:
            _logger.debug("Local pitch keypoint prediction error (%s), attempting fallback", e)

    # --- Mode 2: Roboflow Cloud API Fallback ---
    if not api_key:
        return []

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