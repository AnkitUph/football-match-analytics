import json
import cv2

from apps.matches.pitch_landmarks_32 import ROBOFLOW_KEYPOINTS_32_BY_INDEX

with open("inference_result.json") as f:
    result = json.load(f)

keypoints = result["predictions"][0]["keypoints"]

frame = cv2.imread("calibration_test_frame.jpg")

CONF_THRESHOLD = 0.5

print(f"{'index':>5}  {'x':>7}  {'y':>7}  {'conf':>7}  label")
for kp in keypoints:
    idx = int(kp["class"])  # <-- the fix: use "class" string, not class_id
    conf = kp["confidence"]
    if conf < CONF_THRESHOLD:
        continue

    x, y = int(kp["x"]), int(kp["y"])
    entry = ROBOFLOW_KEYPOINTS_32_BY_INDEX.get(idx)
    label = entry["label"] if entry else "!! NO MAPPING FOUND !!"
    print(f"{idx:>5}  {x:>7}  {y:>7}  {conf:>7.4f}  {label}")

    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(idx), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite("calibration_overlay.jpg", frame)
print("\nSaved calibration_overlay.jpg")
